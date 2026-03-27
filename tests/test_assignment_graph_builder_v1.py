from __future__ import annotations

import gymnasium as gym

import tarware  # noqa: F401
from tarware.warehouse import AgentType

from tarware_ext.controllers import HeuristicController
from tarware_ext.graphs import AssignmentGraphBuilder


def test_assignment_graph_builder_preserves_request_slot_mapping() -> None:
    env = gym.make("tarware-small-2agvs-1pickers-globalobs-v1", disable_env_checker=True)
    try:
        env.reset(seed=21)
        unwrapped = env.unwrapped

        controller = HeuristicController()
        controller.reset(unwrapped, seed=21)

        builder = AssignmentGraphBuilder()
        graph = builder.build(unwrapped, controller=controller)

        request_queue = list(unwrapped.request_queue)
        assert len(graph.task_node_ids) == len(request_queue)
        assert len(graph.task_loc_ids) == len(request_queue)

        slot_to_node = graph.metadata["request_slot_to_node_id"]
        node_to_slot = graph.metadata["node_id_to_request_slot"]

        for slot, item in enumerate(request_queue):
            node_id = int(slot_to_node[slot])
            assert node_id == int(graph.task_node_ids[slot])
            assert int(node_to_slot[node_id]) == slot

            yx = (int(item.y), int(item.x))
            expected_loc = -1
            for loc_id, coords in unwrapped.action_id_to_coords_map.items():
                if (int(coords[0]), int(coords[1])) == yx:
                    expected_loc = int(loc_id)
                    break
            assert int(graph.task_loc_ids[slot]) == expected_loc

        assert graph.action_mask is not None
        assert graph.action_mask.shape == (unwrapped.num_agents, len(request_queue))

        agv_indices = [idx for idx, a in enumerate(unwrapped.agents) if a.type == AgentType.AGV]
        picker_indices = [idx for idx, a in enumerate(unwrapped.agents) if a.type == AgentType.PICKER]
        for idx in agv_indices:
            assert graph.action_mask[idx].dtype == bool
        for idx in picker_indices:
            assert not graph.action_mask[idx].any()
    finally:
        env.close()
