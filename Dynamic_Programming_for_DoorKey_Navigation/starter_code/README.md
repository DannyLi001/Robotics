# DoorKey Navigation using Dynamic Programming

## Directory Structure
```
project_root/
|
├── envs/                   # Environment configuration files
│   ├── known_envs/         # Predefined 5x5, 6x6, and 8x8 maps
│   │   ├── *.env           # Serialized environment states
│   │   └── *.png           # Map layout previews
│   └── random_envs/        # Randomized 10x10 configurations
│       ├── *.png           # Map layout previews
│       └── *.env           # 36 unique random environments
│
├── gif/                    # Generated visualizations
│   └── *.gif               # Solutions for each configuration
│
├── doorkey.py              # Main algorithm implementation
├── utils.py                # Environment utilities
├── create_env.py           # Map generator (pre-configured)
├── requirements.txt        # Dependency list
└── README.md               # This document
```

## Installation
1. Install Python 3.8–3.12
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Execution Commands

### Known Maps (Part A)
Generate solutions for all 7 predefined environments:

```bash
python doorkey.py partA
```

### Random Maps (Part B)
Solve one of the 36 random configurations randomly:

```bash
python doorkey.py partB
```

## Generated Results

### Part A — Known Environments
- **Location:** `gif/`
- **Files:**
  - `doorkey-5x5-normal.gif`: Minimal steps solution for 5x5 map
  - `doorkey-6x6-{normal,direct,shortcut}.gif`: Three 6x6 map variants
  - `doorkey-8x8-{normal,direct,shortcut}.gif`: Three 8x8 map variants

### Part B — Random Environments
- **Location:** `gif/`
- **Files:**
  - `DoorKey-10x10-{1-36}.gif`: Solutions for all 36 random maps

## Visualization Features
- **Agent Motion:** Red triangle shows position and direction of the robot body
- **Key Interaction:** Yellow key disappears when collected
- **Door Unlocking:** Yellow door opens when unlocked
- **Goal Position:** Green spot shows the goal position

---

This README provides complete setup and execution instructions while remaining concise. The directory structure reflects the actual code organization, and GIF descriptions help users interpret the results without prior project knowledge.

