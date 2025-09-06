```mermaid
graph TD

    subgraph "<b>Local Agent Training"
        direction TB
        B("<b>Local Training on Private Data Dₖ</b><br>") --> B1 & B2
        B1["Update <b>fₖ</b> through SSL Loss: <b>VICReg(fₖ(x), fₖ(x'))</b>"]
        B2["Update <b>hₖ</b> Classifier with Loss: <b>CE(hₖ(fₖ(x)), y)</b>"]
    end

    subgraph "<b>Representation Alignment"
        direction TB
        C("<b>Generate Embeddings on Shared Public Data D_pub</b>")
        C --> D["Agent <b>k</b> Computes Embeddings:<b><br>ψ_fₖ = fₖ(x_pub)"]
        D --> E["Combine with Neighbors 
        <b>l ∈ Nₖ:<b><br>ψ̄_fₖ = Σ aₗₖ ψ_fₗ"]
        E --> F["Update Agent Backbone <b>fₖ</b> using <b>Alignment Loss:<br>MSE(ψ_fₖ, ψ̄_fₖ)"]
    end

    subgraph "<b> Classifier Consensus"
        direction TB
        G["<b>Combine Classifier Weights with Neighbors 
        l ∈ Nₖ:</b><br>hₖ ← Σ aₗₖ hₗ"]
    end



    B1 --> C
    B2 --> C
    F --> G

    %% Styling
    style B fill:#bde,stroke:#333,stroke-width:2px,font-size:16px,font-weight:bold
    style C fill:#bde,stroke:#333,stroke-width:2px,font-size:16px,font-weight:bold
    style G fill:#bde,stroke:#333,stroke-width:2px,font-size:16px,font-weight:bold