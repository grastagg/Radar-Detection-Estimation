import pyvista as pv

# Load 3D model
radar_model = pv.read("radar.obj")
uav_model = pv.read("drone.obj")

# Scale and position
radar_model.translate([5, 5, 0])
radar_model.scale([0.1, 0.1, 0.1])

uav_model.translate([-5, -5, 1])
uav_model.scale([0.2, 0.2, 0.2])

# Plot
plotter = pv.Plotter()
plotter.add_mesh(radar_model, color="gray")
plotter.add_mesh(uav_model, color="white")
plotter.show()
