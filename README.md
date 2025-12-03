# 📘 **AI Pathfinding Heuristic Comparison**
### *Advanced Implementation of the A* Search Algorithm on Grid-Based Maps*

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Algorithm](https://img.shields.io/badge/Algorithm-A*%20Pathfinding-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 **Project Overview**

This project implements the **A*** (A-star) pathfinding algorithm **from scratch** on weighted grid maps with a detailed comparison of **three admissible heuristics**:

- 🟦 **Manhattan Distance**
- 🟩 **Euclidean Distance**
- 🟧 **Chebyshev Distance**

The purpose is to demonstrate how heuristic selection impacts:

- Path accuracy
- Search efficiency
- Runtime performance
- Node expansion
- Algorithmic optimality

The algorithm runs through **three complex map layouts**, each with various obstacles and weights, and generates both visualizations and performance summaries.

---

## 🧠 **Features**

✔ Complete A* algorithm implementation  
✔ Efficient priority queue open-set  
✔ 3 interchangeable heuristic functions  
✔ Weighted grid map support  
✔ Complex map layouts  
✔ Automatic path visualizations (PNG)  
✔ Full performance metrics stored in `results/`  
✔ Word report included  
✔ Clean project structure  
✔ Easy to run — no external frameworks needed  

---

## 📂 **Project Structure**

```
AI_Pathfinding_Heuristic_Comparison/
│
├── heuristic_comparison.py        # Main A* implementation with heuristics
│
├── images/                        # Auto-saved path visualizations
│   ├── Map1_WallGap_manhattan.png
│   ├── Map2_WeightedTerrain_euclidean.png
│   └── Map3_MazeBars_chebyshev.png
│
├── results/
│   └── comparison_output.txt      # Summary table of performance
│
└── report/
    └── AI_Pathfinding_Report.docx # Final academic-style report
```

---

## ▶️ **How to Run the Project**

### **1. Install Dependencies**

```
py -m pip install numpy matplotlib
```

### **2. Run the Main Script**

```
py heuristic_comparison.py
```

### This will generate:

- ✔ Terminal summary table
- ✔ PNG visualizations saved in `images/`
- ✔ Comparison summary saved in `results/comparison_output.txt`

---

## 📊 **Sample Output Summary**

```
Map1_WallGap    manhattan   PathLen=20   Cost=19   Expanded=19   Time=0.000257
Map1_WallGap    euclidean   PathLen=20   Cost=19   Expanded=20   Time=0.000218
Map1_WallGap    chebyshev   PathLen=20   Cost=19   Expanded=44   Time=0.000426

Map2_WeightedTerrain manhattan  PathLen=37  Cost=36  Expanded=36  Time=0.000323
Map2_WeightedTerrain euclidean PathLen=36   Cost=35  Expanded=147 Time=0.001151
Map2_WeightedTerrain chebyshev PathLen=36   Cost=35  Expanded=154 Time=0.001451

Map3_MazeBars manhattan   PathLen=77   Cost=76   Expanded=357 Time=0.002796
Map3_MazeBars euclidean  PathLen=77   Cost=76   Expanded=360 Time=0.002272
Map3_MazeBars chebyshev  PathLen=77   Cost=76   Expanded=380 Time=0.002613
```

---

## 🔍 **Heuristic Comparison Insights**

| Heuristic | Strengths | Weaknesses | Best Used When |
|-----------|-----------|------------|----------------|
| **Manhattan** | Fastest, minimal node expansion | Slightly less optimal on weighted maps | 4-direction grids |
| **Euclidean** | Smooth and realistic | More nodes expanded | Weighted terrain |
| **Chebyshev** | Works for diagonal movement | Highest expansion cost | 8-direction movement |

---

## 🏁 **Conclusion**

- **Manhattan** was the fastest and most efficient across all maps.
- **Euclidean** provided the best results on weighted terrain.
- **Chebyshev** expanded more nodes, making it less efficient for these grid layouts.

This project illustrates how heuristic selection plays a crucial role in AI pathfinding efficiency and real-world navigation performance.

---

## 📄 **License**

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute it with attribution.

---
