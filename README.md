# Autonomous Search, Location, and Quantification of Target Plant Species Using Machine Learning and ACO Path Planning

## Motivation
Conventional UAV-based remote sensing utilizes standard coverage algorithms to survey a designated area. These methods typically employ a "lawnmower" pattern, executing a rigid, zig-zag flight path to ensure complete coverage of the terrain. While exhaustive, this approach is highly inefficient when searching for a specific target plant species, as it results in the collection of vast amounts of irrelevant data.


The objective of this project is to overcome this inefficiency by developing a dynamic, learning-based search strategy designed to autonomously locate and quantify target plant species within an a priori unknown environment. The video below shows a photorealistic simulation of an autonomous quadrotor conducting plant search using the Ant Colony Optimization route-planning algorithm. The simulation was conducted using Unreal Engine and MATLAB Simulink.
<figure>
  <video src="media/UE_simu.mp4" width="600" autoplay loop muted playsinline></video>
  <figcaption><em>Video. 1: Photorealistic simulation using Unreal Engine and MATLAB.</em></figcaption>
</figure>

## Methodology
This project presents two distinct approaches for terrain exploration. Instead of scanning blindly, the system utilizes high-level decision-making algorithms to learn the spatial distribution of the environment and prioritize sectors where target discoveries are most probable. The two approaches are:
1. **Reinforcement Learning:** Ant Colony Optimization (ACO) for trajectory generation, guided by a UCB1 algorithm for high-level sector selection.
2. **Supervised Learning:** Ant Colony Optimization (ACO) for trajectory generation, guided by a Multi-Layer Perceptron (MLP) for predictive sector value estimation.

The terrain of interest is modeled as a 2D grid of dimension $I \times J$ shown in Fig. 1 where each cell $s_{ij}$ represents a specific geographic area called a sector. The dimension of the grid is user defined. 
<figure>
  <img src="media/UCB_TS2_Sector.png" width="600">
  <figcaption><em>Fig. 1: Partition of the terrain of interest into sectors.</em></figcaption>
</figure>

The agent selects a target sector $s_{ij}$ to visit from $S \in \mathbb{R}^{I \times J}$  based on the quantity of target plants recorded within that specific sector. An example of target plant species detection during low-altitude flight is shown in Fig. 28, where target plants were detected in sectors 6 and 8. The UCB TPS2 algorithm suggests the center coordinate of the sector to visit to maximize the likelihood of further discoveries. However, DEE-ACS may still be unable to include the coordinate suggested by UCB-TPS2 if the available energy budget is insufficient.







<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>
