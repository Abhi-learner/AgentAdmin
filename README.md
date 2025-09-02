# AgentAdmin Architecture (with Disk Cleanup Flow)

```mermaid
flowchart TD
    Email[Email Ingestion] --> Classifier[LLM Classifier]
    Classifier --> Entities[Entity Extractor]
    Entities --> Intent[Intent Decision Node]

    %% Intent branching
    Intent -->|Disk Cleanup| DiskSubgraph
    Intent -->|Memory/CPU Utilization or Other Task| TaskSubgraph[Task Subgraph Future]
    Intent -->|Unclear| ManualReview[Manual Review] --> End

    %% Disk Cleanup Subgraph
    subgraph DiskSubgraph [Disk Cleanup Flow]
        DiskInfo[Disk Cleanup Info Extractor]
        ActionPlanner[Action Planner]
        Approval[Approval Node]
        Execution[Execution Node]
        End

        DiskInfo --> ActionPlanner
        ActionPlanner -->|Ask Approval| Approval
        ActionPlanner -->|No Approval Required| Execution

        Approval -->|Yes| Execution
        Approval -->|No| End
        Execution --> End
    end
