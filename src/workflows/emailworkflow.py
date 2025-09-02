from langgraph.graph import START , StateGraph, END
from src.state.emailstate import EmailState
from src.nodes.emailnodes import EmailNodes

class EmailGraph(StateGraph):

    def __init__(self):
        self.nodes = EmailNodes()

    def create_graph(self):
        graph = StateGraph(EmailState)
        graph.add_node("classifier", self.nodes.classify_email)
        graph.add_node("extractor", self.nodes.email_entity_extraction)
        graph.add_node("decision_node", self.nodes.decision_intent_node)
        # graph.add_node("policy_extractor", self.nodes.policy_extraction)
        # graph.add_node("policy_eval", self.nodes.policy_eval_node)
        # graph.add_node("decision_gate", self.nodes.decision_gate_node)
        # graph.add_node("approval", approval_node)  # your existing approval node
        # graph.add_node("tool", tool_node)  # your existing tool node
        # graph.add_node("manual_review", self.nodes.manual_review_node)
        # graph.add_node("persistence", persistence_node)

        graph.add_edge(START, "classifier")
        graph.add_edge(START, "extractor")
        graph.add_edge("classifier", "decision_node")
        graph.add_edge("extractor", "decision_node")
        graph.add_node("router", self.nodes.route_node)
        graph.add_node("manual_review", self.nodes.manual_review)

        # Example subgraphs (you’ll define them separately)
        graph.add_node("disk_alert_subgraph", self.nodes.handle_disk_alert)
        graph.add_node("cpu_alert_subgraph", self.nodes.handle_cpu_alert)
        graph.add_node("fs_extension_subgraph", self.nodes.handle_fs_extension)
        graph.add_node("ticket_update_subgraph", self.nodes.handle_ticket_update)
        graph.add_node("memory_alert_subgraph", self.nodes.handle_memory_alert)
        # graph.add_edge("policy_eval", "decision_gate")
        # graph.add_edge("policy_extractor", "policy_eval")
        #
        # def branch_after_gate(state):
        #     act = state["final_decision"]["action"]
        #     return act
        #
        # graph.add_conditional_edges(
        #     "decision_gate",
        #     branch_after_gate,
        #     {
        #         "request_approval": "approval",
        #         "tool": "tool",
        #         "manual_review": "manual_review",
        #     })
        #
        # def branch_after_approval(state):
        #     return "tool" if state.get("approval_result") == "yes" else "persistence"
        #
        # graph.add_conditional_edges(
        #     "approval",
        #     branch_after_approval,
        #     {
        #         "tool": "tool",
        #         "persistence": "persistence",
        #     }
        # )
        #
        # # manual_review → persistence (audit trail)
        # graph.add_edge("manual_review", "persistence")

        # tool → persistence → END
        # graph.add_edge("tool", "persistence")
        graph.add_edge("decision_node", "router")
        graph.add_conditional_edges(
            "router",
            lambda state: state["next_node"],  # this must return the next node name
            {
                "disk_alert_subgraph": "disk_alert_subgraph",
                "cpu_alert_subgraph": "cpu_alert_subgraph",
                "fs_extension_subgraph": "fs_extension_subgraph",
                "ticket_update_subgraph": "ticket_update_subgraph",
                "memory_alert_subgraph": "memory_alert_subgraph",
                "manual_review": "manual_review",
                "end": END,
            }
        )
        graph.add_edge("manual_review", END)
        graph.add_edge("disk_alert_subgraph", END)
        graph.add_edge("cpu_alert_subgraph", END)
        graph.add_edge("fs_extension_subgraph", END)
        graph.add_edge("ticket_update_subgraph", END)
        graph.add_edge("memory_alert_subgraph", END)

        return graph





