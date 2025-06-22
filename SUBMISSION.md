# Motivation for a **Hard Maze** Environment in the Farama Ecosystem

Over nearly two decades the **Hard Maze** task has become the *canonical* benchmark for studying deception and exploration in neuro‑evolution. From the birth of **Novelty Search** to today's quality‑diversity and deep‑neuroevolution methods, Hard Maze has anchored an entire research lineage.

Hard Maze is a navigation problem where an agent must find a path through a deceptive maze where the shortest path to the goal is blocked, forcing exploration of seemingly suboptimal routes. This simple yet powerful task reveals the limitations of traditional fitness functions and emphasizes the need for innovation in search strategies.

Because it is both historically foundational *and* still actively used, a clean, well‑documented **Python** implementation squarely satisfies Farama's criterion of "scientifically important or commonly used" environments.

---

## 1  A Benchmark that Shaped the Field

| Milestone | Contribution | Key Citation |
|-----------|--------------|--------------|
| **Origins & Impact** | Introduced the idea of discarding explicit objectives; validated Novelty Search on Hard Maze. | 1&nbsp;· 3 |
| **Algorithmic Extensions** | ES‑HyperNEAT evolved substrate geometry, demonstrating gains in deceptive domains. | 4&nbsp;· 5 |
| **Divergent‑Search Hybrids** | Coupled novelty with surprise or differential‑evolution operators, again benchmarking on Hard Maze. | 6&nbsp;· 7 |

---

## 2  Continued Relevance in Modern Research

| Modern Theme | Why Hard Maze Matters | Key Citation |
|--------------|----------------------|--------------|
| **Deep Controllers** | *Image Hard Maze* was central to “Deep Neuroevolution,” where GA + NS succeeded while deep RL failed. | 8 |
| **Quality‑Diversity & Evolvability** | Studies on evolvability, sparse‑reward exploration, and visual analysis still default to Hard Maze. | 9&nbsp;· 10&nbsp;· 11 |

---

## 3  Alignment with Farama Guidelines

* **Scientific importance.** Hard Maze is the de‑facto deception benchmark; new divergent‑search algorithms routinely cite results on it.
* **Common use across eras.** From 2008 → 2024 the task appears in ALIFE, GECCO, ECJ, *Artificial Life*, *SN Computer Science*, and high‑impact arXiv pre‑prints.
* **More than a toy.** Minimal geometry, *maximal* deception: standard RL⁄EA baselines still struggle, especially in the pixel variant.
* **Python port fills a gap.** Existing code is fragmented (Matlab / Java / C++). A Gym‑compatible Python environment would unify experimentation, lower replication barriers, and enable curriculum learning or pixel inputs out‑of‑the‑box.

---

## 4  Recommendation

Given its foundational role, proven longevity, and broad methodological relevance, **Hard Maze** clearly merits inclusion in the Farama ecosystem. A polished Python port would become an indispensable resource for researchers investigating exploration, deception, quality‑diversity, and open‑endedness—*exactly* the kinds of scientific questions Farama seeks to support.

---

## 5  References

1.
@inproceedings{Lehman2008ExploitingOT,
  title={Exploiting Open-Endedness to Solve Problems Through the Search for Novelty},
  author={Joel Lehman and Kenneth O. Stanley},
  booktitle={IEEE Symposium on Artificial Life},
  year={2008},
  url={https://api.semanticscholar.org/CorpusID:2367605}
}

2.
@inproceedings{Lehman2008ExploitingOT,
  title={Exploiting Open-Endedness to Solve Problems Through the Search for Novelty},
  author={Joel Lehman and Kenneth O. Stanley},
  booktitle={IEEE Symposium on Artificial Life},
  year={2008},
  url={https://api.semanticscholar.org/CorpusID:2367605}
}

3.
@ARTICLE{6793380,
  author={Lehman, Joel and Stanley, Kenneth O.},
  journal={Evolutionary Computation},
  title={Abandoning Objectives: Evolution Through the Search for Novelty Alone},
  year={2011},
  volume={19},
  number={2},
  pages={189-223},
  keywords={Evolutionary algorithms;deception;novelty search;open-ended evolution;neuroevolution},
  doi={10.1162/EVCO_a_00025}}

4.
  @inproceedings{10.1145/2001576.2001783,
  author = {Risi, Sebastian and Stanley, Kenneth O.},
  title = {Enhancing es-hyperneat to evolve more complex regular neural networks},
  year = {2011},
  isbn = {9781450305570},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/2001576.2001783},
  doi = {10.1145/2001576.2001783},
  abstract = {The recently-introduced evolvable-substrate HyperNEAT algorithm (ES-HyperNEAT) demonstrated that the placement and density of hidden nodes in an artificial neural network can be determined based on implicit information in an infinite-resolution pattern of weights, thereby avoiding the need to evolve explicit placement. However, ES-HyperNEAT is computationally expensive because it must search the entire hypercube, and was shown only to match the performance of the original HyperNEAT in a simple benchmark problem. Iterated ES-HyperNEAT, introduced in this paper, helps to reduce computational costs by focusing the search on a sequence of two-dimensional cross-sections of the hypercube and therefore makes possible searching the hypercube at a finer resolution. A series of experiments and an analysis of the evolved networks show for the first time that iterated ES-HyperNEAT not only matches but outperforms original HyperNEAT in more complex domains because ES-HyperNEAT can evolve networks with limited connectivity, elaborate on existing network structure, and compensate for movement of information within the hypercube.},
  booktitle = {Proceedings of the 13th Annual Conference on Genetic and Evolutionary Computation},
  pages = {1539–1546},
  numpages = {8},
  keywords = {hyperneat, neat, neuroevolution},
  location = {Dublin, Ireland},
  series = {GECCO '11}
  }


5.
  @article{10.1162/ARTL_a_00071,
  author = {Risi, Sebastian and Stanley, Kenneth O.},
  title = {An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density, and Connectivity of Neurons},
  journal = {Artificial Life},
  volume = {18},
  number = {4},
  pages = {331-363},
  year = {2012},
  month = {10},
  abstract = {Intelligence in nature is the product of living brains, which are themselves the product of natural evolution. Although researchers in the field of neuroevolution (NE) attempt to recapitulate this process, artificial neural networks (ANNs) so far evolved through NE algorithms do not match the distinctive capabilities of biological brains. The recently introduced hypercube-based neuroevolution of augmenting topologies (HyperNEAT) approach narrowed this gap by demonstrating that the pattern of weights across the connectivity of an ANN can be generated as a function of its geometry, thereby allowing large ANNs to be evolved for high-dimensional problems. Yet the positions and number of the neurons connected through this approach must be decided a priori by the user and, unlike in living brains, cannot change during evolution. Evolvable-substrate HyperNEAT (ES-HyperNEAT), introduced in this article, addresses this
                  limitation by automatically deducing the node geometry from implicit information in the pattern of weights encoded by HyperNEAT, thereby avoiding the need to evolve explicit placement. This approach not only can evolve the location of every neuron in the network, but also can represent regions of varying density, which means resolution can increase holistically over evolution. ES-HyperNEAT is demonstrated through multi-task, maze navigation, and modular retina domains, revealing that the ANNs generated by this new approach assume natural properties such as neural topography and geometric regularity. Also importantly, ES-HyperNEAT's compact indirect encoding can be seeded to begin with a bias toward a desired class of ANN topographies, which facilitates the evolutionary search. The main conclusion is that ES-HyperNEAT significantly expands the scope of neural structures that evolution can discover.},
  issn = {1064-5462},
  doi = {10.1162/ARTL_a_00071},
  url = {https://doi.org/10.1162/ARTL\_a\_00071},
  eprint = {https://direct.mit.edu/artl/article-pdf/18/4/331/1663121/artl\_a\_00071.pdf},
}

6.
@inproceedings{10.1145/3071178.3071179,
author = {Gravina, Daniele and Liapis, Antonios and Yannakakis, Georgios N.},
title = {Coupling novelty and surprise for evolutionary divergence},
year = {2017},
isbn = {9781450349208},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3071178.3071179},
doi = {10.1145/3071178.3071179},
abstract = {Divergent search techniques applied to evolutionary computation, such as novelty search and surprise search, have demonstrated their efficacy in highly deceptive problems compared to traditional objective-based fitness evolutionary processes. While novelty search rewards unseen solutions, surprise search rewards unexpected solutions. As a result these two algorithms perform a different form of search since an expected solution can be novel while an already seen solution can be surprising. As novelty and surprise search have already shown much promise individually, the hypothesis is that an evolutionary process that rewards both novel and surprising solutions will be able to handle deception in a better fashion and lead to more successful solutions faster. In this paper we introduce an algorithm that realises both novelty and surprise search and we compare it against the two algorithms that compose it in a number of robot navigation tasks. The key findings of this paper suggest that coupling novelty and surprise is advantageous compared to each search approach on its own. The introduced algorithm breaks new ground in divergent search as it outperforms both novelty and surprise in terms of efficiency and robustness, and it explores the behavioural space more extensively.},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
pages = {107–114},
numpages = {8},
keywords = {NEAT, deception, divergent search, fitness-based evolution, map navigation, novelty search, surprise search},
location = {Berlin, Germany},
series = {GECCO '17}
}

7.
@inproceedings{inproceedings,
author = {Mason, Karl and Howley, Enda},
year = {2018},
month = {07},
pages = {},
title = {Maze Navigation using Neural Networks Evolved with Novelty Search and Differential Evolution}
}

8.
@misc{
such2019deep,
title={Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning},
author={Felipe Petroski Such and Vashisht Madhavan and Edoardo Conti and Joel Lehman and Kenneth O. Stanley and Jeff Clune},
year={2019},
url={https://openreview.net/forum?id=HyGh4sR9YQ},
}

9.
@misc{doncieux2020noveltysearchmakesevolvability,
      title={Novelty Search makes Evolvability Inevitable},
      author={Stephane Doncieux and Giuseppe Paolo and Alban Laflaquière and Alexandre Coninx},
      year={2020},
      eprint={2005.06224},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2005.06224},
}

10.
@article{DBLP:journals/corr/abs-2102-03140,
  author       = {Giuseppe Paolo and
                  Alexandre Coninx and
                  St{\'{e}}phane Doncieux and
                  Alban Laflaqui{\`{e}}re},
  title        = {Sparse Reward Exploration via Novelty Search and Emitters},
  journal      = {CoRR},
  volume       = {abs/2102.03140},
  year         = {2021},
  url          = {https://arxiv.org/abs/2102.03140},
  eprinttype    = {arXiv},
  eprint       = {2102.03140},
  timestamp    = {Thu, 14 Oct 2021 09:18:20 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2102-03140.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}


11.
@article{10.1007/s42979-022-01064-6,
author = {Sarti, Stefano and Adair, Jason and Ochoa, Gabriela},
title = {Recombination and Novelty in Neuroevolution: A Visual Analysis},
year = {2022},
issue_date = {May 2022},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
volume = {3},
number = {3},
url = {https://doi.org/10.1007/s42979-022-01064-6},
doi = {10.1007/s42979-022-01064-6},
abstract = {Neuroevolution has re-emerged as an active topic in the last few years. However, there is a lack of accessible tools to analyse, contrast and visualise the behaviour of neuroevolution systems. A variety of search strategies have been proposed such as Novelty search and Quality-Diversity search, but their impact on the evolutionary dynamics is not well understood. We propose using a data-driven, graph-based model, search trajectory networks (STNs) to analyse, visualise and directly contrast the behaviour of different neuroevolution search methods. Our analysis uses NEAT for solving maze problems with two search strategies: novelty-based and fitness-based, and including and excluding the crossover operator. We model and visualise the trajectories, contrasting and illuminating the behaviour of the studied neuroevolution variants. Our results confirm the advantages of novelty search in this setting, but challenge the usefulness of recombination.},
journal = {SN Comput. Sci.},
month = mar,
numpages = {15},
keywords = {Neuroevolution, NEAT, Algorithm analysis, Complex networks, Search trajectory networks, Novelty search, Recombination}
}
