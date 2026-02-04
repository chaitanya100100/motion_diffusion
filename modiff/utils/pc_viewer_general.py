import time
import numpy as np
import viser
import viser.transforms as tf
from typing import Optional, List, Union, Dict, Any

from .torch_utils import to_numpy


class PointCloudViewer:
    """
    Generalized viewer for static and dynamic 3D data.
    """

    def __init__(self, port: int = 8080):
        # --- Server Setup ---
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")

        # --- Data Storage ---
        self._visualizables: Dict[str, Dict[str, Any]] = {}
        self.num_frames = 0

        # --- GUI State ---
        self._setup_global_gui()

    # ================================================================
    # Public API for Adding Data
    # ================================================================

    def add_point_cloud(
        self,
        name: str,
        points: Union[np.ndarray, List[np.ndarray]],
        colors: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        conf: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        point_size: float = 0.005,
        static: bool = False,
    ):
        points = self._standardize_input(points, static)
        colors = self._standardize_input(colors, static)
        conf = self._standardize_input(conf, static)

        self._visualizables[name] = {
            "type": "point_cloud",
            "static": static,
            "points": points,
            "colors": colors,
            "conf": conf,
            "point_size": point_size,
            "handles": [],
            "frame_nodes": [],
        }

    def add_mesh(
        self,
        name: str,
        vertices: Union[np.ndarray, List[np.ndarray]],
        faces: np.ndarray,
        color: tuple = (0.7, 0.8, 1.0),
        static: bool = False,
    ):
        vertices = self._standardize_input(vertices, static)
        faces = self._standardize_input(faces, True)

        self._visualizables[name] = {
            "type": "mesh",
            "static": static,
            "vertices": vertices,
            "faces": faces,
            "color": color,
            "handles": [],
            "frame_nodes": [],
        }

    def add_trajectory(
        self,
        name: str,
        poses: Union[np.ndarray, List[np.ndarray]],
        axes_length: float = 0.15,
        color: tuple = (236, 0, 236),
    ):
        static = False  # Trajectories are always dynamic
        poses = self._standardize_input(poses, static)
        self._visualizables[name] = {
            "type": "trajectory",
            "static": static,
            "poses": poses,
            "axes_length": axes_length,
            "origin_color": color,
            "handles": [],
            "frame_nodes": [],
        }

    def add_camera(
        self,
        name: str,
        extrinsics: Union[np.ndarray, List[np.ndarray]],
        intrinsics: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        scale: float = 0.1,
        color: tuple = (50, 205, 50),
        static: bool = False,
    ):
        # Static cameras: handle dimension expansion for single camera input
        extrinsics = self._standardize_input(extrinsics, static=static)
        intrinsics = self._standardize_input(intrinsics, static=static)
        images = self._standardize_input(images, static=static)

        if static and extrinsics.ndim == 2:
            extrinsics = extrinsics[None, ...]
            intrinsics = intrinsics[None, ...] if intrinsics is not None else None
            images = images[None, ...] if images is not None else None

        self._visualizables[name] = {
            "type": "camera",
            "static": static,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "scale": scale,
            "color": color,
            "images": images,
            "handles": [],
            "frame_nodes": [],
        }

    # ================================================================
    # Internal Data Handling
    # ================================================================

    def _standardize_input(self, data, static):
        if data is None:
            return data
        data = to_numpy(data)
        if not static:
            num_frames = len(data)
            if num_frames > self.num_frames:
                self.num_frames = num_frames
        return data

    # ================================================================
    # Rendering & GUI
    # ================================================================

    def _setup_global_gui(self):
        """Global controls separate from specific objects."""
        with self.server.add_gui_folder("Global Settings"):
            self.gui_reset_up = self.server.add_gui_button("Reset Up Direction")

        @self.gui_reset_up.on_click
        def _(event):
            client = event.client
            if client:
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

    def _create_item_gui(self, name, item):
        # Create a separate folder for each visualizable item
        item_folder = self.server.add_gui_folder(name)

        with item_folder:
            # A. Visibility: show
            cb = self.server.add_gui_checkbox("show", True)
            item["gui_visible"] = cb

            # For static items, toggle handles directly.For dynamic items, the loop in run() reads this checkbox value
            @cb.on_update
            def _(e, item=item):
                if item["static"]:
                    for h in item["handles"]:
                        h.visible = e.target.value

            # B. Point Size (if applicable): point size
            if item["type"] == "point_cloud":
                sl = self.server.add_gui_slider("point size", 0.001, 0.05, 0.001, item["point_size"])
                item["gui_size_slider"] = sl

                @sl.on_update
                def _(e, item=item):
                    for h in item["handles"]:
                        h.point_size = e.target.value

                # Confidence Slider (Only if conf data is present)
                if item["conf"] is not None:
                    sl_conf = self.server.add_gui_slider("top conf %", 0, 100, 1, 100)

                    @sl_conf.on_update
                    def _(e, item=item):
                        if item["static"]:
                            self._filter_points(item, e.target.value, t=None)
                        else:
                            for t in range(len(item["points"])):
                                self._filter_points(item, e.target.value, t=t)

            # C. Time Mode (if dynamic): time mode
            if not item["static"] and self.num_frames > 0:
                mode = self.server.add_gui_button_group("time mode", ("Standard", "Cumulative"))
                item["gui_mode"] = mode

    def _render_all(self, names_to_render=None):
        """Generates the Viser handles and centralized GUI."""

        for name, item in self._visualizables.items():
            if names_to_render is not None and name not in names_to_render:
                continue

            if "gui_visible" not in item:
                self._create_item_gui(name, item)

            # Render Geometry
            if item["static"]:
                self._render_element(f"/{name}", item, t=None)
            else:
                item["frame_nodes"] = []
                for t in range(self.num_frames):
                    frame_path = f"/{name}/{t}"
                    frame_node = self.server.add_frame(frame_path, show_axes=False)
                    item["frame_nodes"].append(frame_node)

                    # Check if this object has data at time t
                    data_len = self._get_data_length(item)
                    if t < data_len:
                        self._render_element(f"{frame_path}/geo", item, t=t)

    def _render_element(self, node_path, item, t=None):
        """Render a single element, either static (t=None) or at a specific t."""
        if item["type"] == "point_cloud":
            self._render_point_cloud(node_path, item, t)

        elif item["type"] == "mesh":
            self._render_mesh(node_path, item, t)

        elif item["type"] == "trajectory":
            self._render_trajectory_frame(node_path, item, t)

        elif item["type"] == "camera":
            self._render_camera(node_path, item, t)

        else:
            raise ValueError(f"Unsupported element type: {item['type']}")

    def _get_data_length(self, item):
        """Get the number of frames for a dynamic item."""
        if "points" in item:
            return len(item["points"])
        elif "poses" in item:
            return len(item["poses"])
        elif "vertices" in item:
            return len(item["vertices"])
        elif "extrinsics" in item:
            return len(item["extrinsics"])
        return 0

    # ================================================================
    # Type-Specific Rendering Methods
    # ================================================================

    def _render_point_cloud(self, path, item, t=None):
        """Render a point cloud at the given path."""

        points = item["points"] if t is None else item["points"][t]
        points = points.reshape(-1, 3)

        if item["colors"] is None:
            colors = (0.7, 0.7, 0.7)
        else:
            colors = item["colors"] if t is None else item["colors"][t]
            colors = colors.reshape(-1, 3)

        handle = self.server.add_point_cloud(path, points=points, colors=colors, point_size=item["point_size"])
        item["handles"].append(handle)

    def _filter_points(self, item, percentage, t=None):
        """Helper to filter point cloud arrays based on confidence percentile."""
        points = item["points"] if t is None else item["points"][t]
        colors = item["colors"] if t is None else item["colors"][t]
        conf = item["conf"] if t is None else item["conf"][t]
        handle = item["handles"][0] if t is None else item["handles"][t]
        if conf is None or percentage >= 100:
            handle.points = points
            handle.colors = colors
            return
        threshold = np.percentile(conf, 100 - percentage)
        mask = conf >= threshold
        if isinstance(colors, np.ndarray) and colors.shape == points.shape:
            colors = colors[mask]
        points = points[mask]
        handle.points = points
        handle.colors = colors
        return

    def _render_mesh(self, path, item, t=None):
        """Render a mesh at the given path."""
        # Extract vertices based on whether this is static or dynamic
        vertices = item["vertices"] if t is None else item["vertices"][t]
        faces = item["faces"]
        color = item["color"]

        # if len(color) == len(vertices):
        #     import trimesh
        #     print("ADDING TRIMESH MESH")
        #     trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=color)
        #     handle = self.server.add_mesh_trimesh(path, trimesh_mesh)
        #     item["handles"].append(handle)
        #     return
        handle = self.server.add_mesh_simple(path, vertices=vertices, faces=faces, color=color)
        item["handles"].append(handle)

    def _render_trajectory_frame(self, path, item, t=None):
        """Render a single trajectory frame (coordinate system) at the given path."""
        assert t is not None, "Trajectories are always dynamic and require a t"

        pose = item["poses"][t]
        R, p = pose[:3, :3], pose[:3, 3]
        handle = self.server.add_frame(
            path,
            wxyz=tf.SO3.from_matrix(R).wxyz,
            position=p,
            axes_length=item["axes_length"],
            axes_radius=0.01,
            origin_color=item["origin_color"],
        )
        item["handles"].append(handle)

    def _render_camera(self, path, item, t=None):
        """Render a camera frustum at the given path."""
        # Determine which indices to render (static: all, dynamic: current frame)
        idxs = list(range(len(item["extrinsics"]))) if item["static"] else [t]

        for idx in idxs:
            pose = item["extrinsics"][idx]
            intrinsics = item["intrinsics"][idx] if item["intrinsics"] is not None else None
            image = item["images"][idx] if item["images"] is not None else None

            fov, aspect = self._get_fov_aspect(intrinsics)
            R, p = pose[:3, :3], pose[:3, 3]
            # R_cv_to_gl = np.diag([1, -1, -1])
            # R = R @ R_cv_to_gl
            handle = self.server.add_camera_frustum(
                path,
                fov=fov,
                aspect=aspect,
                scale=item["scale"],
                wxyz=tf.SO3.from_matrix(R).wxyz,
                position=p,
                color=item["color"],
                image=image,
            )
            item["handles"].append(handle)

    def _get_fov_aspect(self, K):
        if K is None:
            K = np.array([[150.0 / 512, 0.0, 255.5 / 512], [0.0, 150.0 / 512, 255.5 / 512], [0.0, 0.0, 1.0]])
        focal = (K[0, 0] + K[1, 1]) / 2
        fov = 2 * np.arctan(K[0, 2] / focal)
        aspect = K[0, 2] / K[1, 2]
        return fov, aspect

    # ================================================================
    # Animation Loop
    # ================================================================

    def run(self):
        self._render_all()

        if self.num_frames == 0:
            print("No temporal data added. Running static server.")
            while True:
                time.sleep(1.0)

        with self.server.add_gui_folder("Playback"):
            self.gui_t = self.server.add_gui_slider("Frame", 0, self.num_frames - 1, 1, 0)
            self.gui_play = self.server.add_gui_checkbox("Playing", True)
            self.gui_fps = self.server.add_gui_slider("FPS", 1, 60, 1, 10)

        while True:
            if self.gui_play.value:
                new_t = (self.gui_t.value + 1) % self.num_frames
                self.gui_t.value = new_t

            current_t = self.gui_t.value

            # Update Visibility for Dynamic Elements
            for name, item in self._visualizables.items():
                if item["static"]:
                    continue

                global_show = item["gui_visible"].value
                is_cumulative = item["gui_mode"].value == "Cumulative"
                for t, node in enumerate(item["frame_nodes"]):
                    if not global_show:
                        node.visible = False
                        continue

                    if is_cumulative:
                        node.visible = t <= current_t
                    else:
                        node.visible = t == current_t

            time.sleep(1.0 / self.gui_fps.value)


def visualize_results(
    pc_list=None,
    color_list=None,
    conf_list=None,
    intrinsics_list=None,
    extrinsics_list=None,
    traj_list=None,
    verts_list=None,
    faces=None,
    static_pc=None,
    static_color=None,
    port=8080,
):
    viewer = PointCloudViewer(port=port)
    if pc_list is not None:
        viewer.add_point_cloud("pc", pc_list, colors=color_list, conf=conf_list, static=False)

    if extrinsics_list is not None:
        viewer.add_camera("camera", extrinsics_list, intrinsics_list, images=color_list, static=False)
        viewer.add_trajectory("camera traj", extrinsics_list, color=(0, 0, 0))
    if traj_list is not None:
        viewer.add_trajectory("traj", traj_list)
    if verts_list is not None:
        viewer.add_mesh("mesh", verts_list, faces, static=False)
    if static_pc is not None:
        viewer.add_point_cloud("static_pc", static_pc, colors=static_color, static=True)
    viewer.run()


if __name__ == "__main__":
    viewer = PointCloudViewer(port=8080)
    # Example usage
    # viewer.add_point_cloud("Input Cloud", ...)
    # viewer.run()
