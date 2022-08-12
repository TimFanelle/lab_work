import mujoco_py

activations = [[0.7, 0.3, 0.3]*4]*2000
kinematics = []


model = mujoco_py.load_model_from_path("quadruped.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
for u in activations:
    sim.data.ctrl[:] = u
    sim.step()
    kinematics.append(sim.data.qpos[1:])
    viewer.render()