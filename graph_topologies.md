```mermaid
graph TB
    subgraph Fully Connected
        direction LR
        A((1)) --- B((2))
        A --- C((3))
        A --- D((4))
        A --- E((5))
        B --- C
        B --- D
        B --- E
        C --- D
        C --- E
        D --- E
    end

    subgraph Ring
        %% Nodes and connections are defined to encourage a more circular layout
        H((3))
        G((2)) --- H --- I((4))
        F((1)) --- G
        J((5)) --- I
        F --- J
    end

    subgraph Random
        direction LR
        K((1)) --- L((2))
        L --- M((3))
        M --- N((4))
        K --- N
        L --- O((5))
    end

    subgraph Isolated
        direction LR
        P((1))
        Q((2))
        R((3))
        S((4))
        T((5))
    end

    style A fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style B fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style C fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style D fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style E fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style F fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style G fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style H fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style I fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style J fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style K fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style L fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style M fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style N fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style O fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style P fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style Q fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style R fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style S fill:#000080,color:#fff,stroke:#333,stroke-width:2px
    style T fill:#000080,color:#fff,stroke:#333,stroke-width:2px

