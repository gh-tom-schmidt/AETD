# AETD
# 🚛 AETD – Autonomous Euro Truck Driver

> *“Let the AI take the wheel and deliver whatever’s in that trailer.”*

---

<p align="center"> <a href="./LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License"></a> <img src="https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg" alt="Python"> <img src="https://img.shields.io/badge/Status-Alpha-orange.svg" alt="Status"> <img src="https://img.shields.io/badge/Platform-Euro%20Truck%20Simulator%202-lightgrey.svg" alt="Platform"> <img src="https://img.shields.io/github/last-commit/gh-tom-schmidt/AETD" alt="Last Commit"> <img src="https://img.shields.io/github/issues/gh-tom-schmidt/AETD" alt="Issues"> <img src="https://img.shields.io/github/stars/gh-tom-schmidt/AETD?style=social" alt="Stars"> <img src="https://img.shields.io/github/forks/gh-tom-schmidt/AETD?style=social" alt="Forks"> </p>

<p align="center"> <img src="docs/images/HomeImageET.png" alt="AETD" width="600"/> </p>

## 🌍 Overview

I’ve always found it fascinating to watch cars drive on their own. In the real world, this is a challenging task that requires:

* ⚙️ Extensive testing
* 🧩 Handling edge cases
* 🏛️ Navigating bureaucracy
* 💰 A lot of funding

Fortunately, simulators mimic real-world driving without the high cost—when something goes wrong, you simply restart.

But most simulators are **dull**. Luckily, some games are both fun *and* realistic enough to enhance with an AI driver.

One of the best examples: **Euro Truck Simulator 2** 🚚.
It not only provides a **realistic driving experience**, but also gives a **clear and meaningful objective**: deliver cargo from point A to point B.

This is where **AETD** begins—an AI trucker with one simple, purposeful mission:
**Autonomously drive and deliver the load.**

---

## ⚠️ Disclaimer

This project is **heavily under construction** and currently in **alpha state**.
Due to limited hardware and funding, the AI Driver is not fully implemented or tested.

---

## 🚀 Getting Started

### 1️⃣ Install requirements

You can use the provided setup script:

```bash
bash install.sh
```

### 2️⃣ Run the AI driver

```bash
python start.py
```

That’s it—the AI will attempt to take the wheel!

---

## 📂 Main Project Structure

```
AETD/
│
├── modules/     # Core AI & analysis (direction, speed extraction, road segmentation, etc.)
├── driver/      # Handles input/output (keyboard control, screen recording, etc.)
├── safety/      # Safety layer that prevents risky moves & oversteering
├── debug/       # GUI system for testing & debugging modules (run with debug.py)
├── configs/     # Global config system (.conf files, defaults + overrides)
│
├── start.py     # Main entry point to run the AI driver
├── debug.py     # Entry point for debugging
└── install.sh   # Setup script
```

## ⚙️ Configuration System

AETD uses a **global configuration system** that makes it easy to manage parameters across the entire project.
This way you can adjust **how the AI behaves, what modules are used, and how data is processed** — all without digging into the code.

### 🔑 How it works

1. **Default config** (`default.conf`)

   * Always loaded first.
   * Contains the baseline values used across the whole project.
   * Ensures that the project runs out-of-the-box, even if you don’t provide your own configs.

2. **Custom config(s)** (e.g., `myconfig.conf`)

   * Loaded *on top* of the default.
   * Only needs to contain the values you want to change.
   * Any parameters not specified will fall back to the default values.

3. **Global access**

   * Once loaded, configs are accessible globally across modules, drivers, debug tools, and safety systems.
   * This ensures consistency (e.g., the same FPS limit, resolution, or steering sensitivity everywhere).

---

## 📊 Resources

* 📂 **Datasets, models & resources** → [arcxyon.com](https://arcxyon.com/)
* 📒 **Notebooks** → Available on [Kaggle](https://www.kaggle.com/tomschmidt25)
* 📜 **Scripts** → Included for automatic downloads and setup

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**.