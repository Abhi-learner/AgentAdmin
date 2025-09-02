from langgraph.graph import START , StateGraph, END
from src.state.diskalertstate import DiskAlertState
from src.nodes.diskalertnodes import DiskAlertNodes

class DiskGraph(StateGraph):

    def __init__(self):
        self.nodes = DiskAlertNodes()

    def create_disk_alert_graph(self):
        graph = StateGraph(DiskAlertState)
        graph.add_node("extract_disk_alert_info", self.nodes.extract_info)
        graph.add_node("find_large_files", self.nodes.discover_files)
        graph.add_node("cleanup_planner_node", self.nodes.cleanup_planner_node)
        graph.add_node("approval_node", self.nodes.approval_node)
        graph.add_node("execution_node", self.nodes.execution_node)
        graph.add_edge(START, "extract_disk_alert_info")
        graph.add_edge("extract_disk_alert_info", "find_large_files")
        graph.add_edge("find_large_files", "cleanup_planner_node")

        def planner_route(state: DiskAlertState) -> str:
            action = state.get("cleanup_plan", {}).get("action")
            if action == "execute":
                return "execution_node"
            elif action == "request_approval":
                return "approval_node"
            return END
        graph.add_conditional_edges("cleanup_planner_node", planner_route, ["execution_node", "approval_node", END])

        def approval_route(state: DiskAlertState) -> str:
            if state.get("approval_result") == "approved":
                return "execution_node"
            else:
                return END

        graph.add_conditional_edges("approval_node", approval_route, ["execution_node", END])
        graph.add_edge("execution_node", END)
        return graph