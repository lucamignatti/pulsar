// Pybind11 bindings — obs builder, action parser, goal vectors.
// C++ is the single source of truth; this module exposes the same logic
// used by the training system so the viz script never duplicates it.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulsar/pulsar_env.hpp"

namespace py = pybind11;

static pulsar::Vec3 to_vec3(const py::array_t<float>& a) {
  auto r = a.unchecked<1>();
  return {r(0), r(1), r(2)};
}

PYBIND11_MODULE(pulsar_py, m) {
  m.doc() = "Pulsar C++ bindings: obs builder, action parser, goal vectors.";

  // ---- build_obs -----------------------------------------------------------
  // Returns a 47-element float32 array for one agent.
  // Pass invert=True for the Orange agent (index 1).
  // ego / opp args are the respective car states; ball is shared.
  m.def(
      "build_obs",
      [](bool invert,
         py::array_t<float> ego_pos, py::array_t<float> ego_vel,
         py::array_t<float> ego_fwd, py::array_t<float> ego_up,
         py::array_t<float> ego_ang_vel, float ego_boost,
         py::array_t<float> ball_pos, py::array_t<float> ball_vel,
         py::array_t<float> ball_ang_vel,
         py::array_t<float> opp_pos, py::array_t<float> opp_vel,
         py::array_t<float> opp_fwd, py::array_t<float> opp_up,
         py::array_t<float> opp_ang_vel, float opp_boost)
          -> py::array_t<float> {
        pulsar::CarState ego{};
        ego.position         = to_vec3(ego_pos);
        ego.velocity         = to_vec3(ego_vel);
        ego.forward          = to_vec3(ego_fwd);
        ego.up               = to_vec3(ego_up);
        ego.angular_velocity = to_vec3(ego_ang_vel);
        ego.boost            = ego_boost;

        pulsar::CarState opp{};
        opp.position         = to_vec3(opp_pos);
        opp.velocity         = to_vec3(opp_vel);
        opp.forward          = to_vec3(opp_fwd);
        opp.up               = to_vec3(opp_up);
        opp.angular_velocity = to_vec3(opp_ang_vel);
        opp.boost            = opp_boost;

        pulsar::BallState ball{};
        ball.position         = to_vec3(ball_pos);
        ball.velocity         = to_vec3(ball_vel);
        ball.angular_velocity = to_vec3(ball_ang_vel);

        auto result = py::array_t<float>(47);
        float* out = result.mutable_data();
        pulsar::detail::write_car_obs(out, ego,  invert);
        pulsar::detail::write_ball_obs(out, ball, invert);
        pulsar::detail::write_car_obs(out, opp,  invert);
        return result;
      },
      py::arg("invert"),
      py::arg("ego_pos"), py::arg("ego_vel"),
      py::arg("ego_fwd"), py::arg("ego_up"),
      py::arg("ego_ang_vel"), py::arg("ego_boost"),
      py::arg("ball_pos"), py::arg("ball_vel"), py::arg("ball_ang_vel"),
      py::arg("opp_pos"), py::arg("opp_vel"),
      py::arg("opp_fwd"), py::arg("opp_up"),
      py::arg("opp_ang_vel"), py::arg("opp_boost"),
      "Build 47-dim obs array. invert=True for Orange (agent_id=1).");

  // ---- parse_action --------------------------------------------------------
  // Returns a dict with throttle/steer/pitch/yaw/roll (float) and
  // jump/boost/handbrake (bool).  agent_id=1 mirrors steer/yaw/roll.
  m.def(
      "parse_action",
      [](py::array_t<float> values, int agent_id) -> py::dict {
        auto r = values.unchecked<1>();
        float vals[8];
        for (int i = 0; i < 8; ++i) vals[i] = r(i);
        const auto ctrl =
            pulsar::PulsarContinuousActionParser::parse(vals, agent_id);
        py::dict d;
        d["throttle"]  = ctrl.throttle;
        d["steer"]     = ctrl.steer;
        d["pitch"]     = ctrl.pitch;
        d["yaw"]       = ctrl.yaw;
        d["roll"]      = ctrl.roll;
        d["jump"]      = ctrl.jump;
        d["boost"]     = ctrl.boost;
        d["handbrake"] = ctrl.handbrake;
        return d;
      },
      py::arg("values"), py::arg("agent_id"),
      "Parse 8-dim float action → controller dict. agent_id=1 mirrors Orange.");

  // ---- goal vectors --------------------------------------------------------
  m.def(
      "goal_score",
      []() -> py::array_t<float> {
        auto a = py::array_t<float>(6);
        float* p = a.mutable_data();
        p[0] = 0.f; p[1] = 1.f; p[2] = 321.f / 2000.f;
        p[3] = 0.f; p[4] = 0.f; p[5] = 0.f;
        return a;
      },
      "Goal vector for the scoring (Blue/even) agent: [0, 1, 0.1605, 0, 0, 0]");

  m.def(
      "goal_defend",
      []() -> py::array_t<float> {
        auto a = py::array_t<float>(6);
        float* p = a.mutable_data();
        p[0] = 0.f; p[1] = -1.f; p[2] = 321.f / 2000.f;
        p[3] = 0.f; p[4] =  0.f; p[5] = 0.f;
        return a;
      },
      "Goal vector for the defending (Orange/odd) agent: [0, -1, 0.1605, 0, 0, 0]");
}
