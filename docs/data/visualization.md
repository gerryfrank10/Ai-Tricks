# Data Visualization in Python

Data visualization is a key component in data analysis to communicate insights effectively. Python provides a rich ecosystem of visualization libraries like **Matplotlib**, **Seaborn**, **pandas Plotting**, and **Bokeh**.

---

## 1. Tools

### 1.1 **Matplotlib**
- The most widely used low-level library for creating static, animated, and interactive plots.
  ```bash
  pip install matplotlib
  ```

### 1.2 **Seaborn**
- Built on top of Matplotlib, offering high-level functions for attractive statistical graphics.
  ```bash
  pip install seaborn
  ```

### 1.3 **pandas Plotting**
- Provides an easy interface to plot data from DataFrames or Series.
  ```bash
  pip install pandas
  ```

### 1.4 **Bokeh**
- Used for building interactive, web-ready plots.
  ```bash
  pip install bokeh
  ```

---

## 2. Anatomy of a Figure

A **Matplotlib figure** consists of several components:
1. **Figure**: The entire plotting area.
2. **Axes**: A single plot or graph (can have multiple axes in one figure).
3. **Data**: The actual data points to be plotted.
4. **Labels/Titles**: Text annotations.
5. **Legend**: Explains elements of the plot.

Example:
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # Create a figure and axis
ax.plot([0, 1, 2], [0, 1, 4])  # Plot data
ax.set_title("Example Plot")  # Add title
ax.set_xlabel("X-axis")       # Label X-axis
ax.set_ylabel("Y-axis")       # Label Y-axis
plt.show()
```

---

## 3. Subplots Layout

Subplots allow you to display multiple plots on the same figure.

### 3.1 Creating Multiple Axes
```python
fig, axes = plt.subplots(2, 2)  # Create a 2x2 grid of subplots
axes[0, 0].plot([1, 2, 3], [1, 4, 9], 'r')  # Top-left
axes[0, 1].scatter([1, 2, 3], [1, 4, 9])    # Top-right
axes[1, 0].bar([1, 2, 3], [1, 4, 9])        # Bottom-left
axes[1, 1].pie([10, 20, 30])                # Bottom-right
plt.tight_layout()
plt.show()
```

---

## 4. Basic Plots

### 4.1 Line Plot
Useful for visualizing trends over time or continuous data.
```python
plt.plot([0, 1, 2], [0, 1, 4])  # x and y data
plt.title("Line Plot")
plt.show()
```

### 4.2 Scatter Plot
Shows relationships between two variables.
```python
plt.scatter([1, 2, 3], [4, 5, 6], c='red')
plt.title("Scatter Plot")
plt.show()
```

### 4.3 Bar Plot
For comparing categories.
```python
plt.bar(["A", "B", "C"], [4, 7, 2], color='green')
plt.title("Bar Plot")
plt.show()
```

### 4.4 Pie Chart
Visualize proportions.
```python
plt.pie([15, 30, 45, 10], labels=["A", "B", "C", "D"], autopct="%1.1f%%")
plt.title("Pie Chart")
plt.show()
```

### 4.5 Contour Plot
Displays contour lines for 3D height data.
```python
import numpy as np

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
plt.contour(X, Y, Z, levels=20, cmap="viridis")
plt.title("Contour Plot")
plt.show()
```

---

## 5. Scales

### Logarithmic Scales
```python
plt.plot([0.1, 1, 10, 100], [1, 10, 100, 1000])
plt.xscale('log')
plt.yscale('log')
plt.title("Logarithmic Scale")
plt.show()
```

---

## 6. Projections

3D plotting can be done using the `mpl_toolkits.mplot3d` module.

```python
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([1, 2, 3], [1, 4, 9], [10, 20, 30])  # x, y, z
ax.set_title("3D Scatter Plot")
plt.show()
```

---

## 7. Colors and Colormaps

### Custom Colors
```python
plt.plot([0, 1, 2], [0, 1, 4], color='springgreen')
plt.show()
```

### Colormaps
Colormaps are useful for translating numerical data into colors.
```python
import numpy as np
data = np.random.rand(10, 10)
plt.imshow(data, cmap='coolwarm', interpolation='nearest')
plt.colorbar()  # Add a color key
plt.show()
```

---

## 8. Markers and Styles

### Markers
Used to highlight points.
```python
plt.plot([1, 2, 3], [1, 4, 9], marker='o', linestyle='--')
plt.show()
```

### Styles
Apply predefined styles.
```python
plt.style.use('ggplot')
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
```

---

## 9. Animation

Matplotlib supports animations for dynamic and evolving plots.
```python
from matplotlib.animation import FuncAnimation

x, y = [], []
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    return line,

def update(frame):
    x.append(frame)
    y.append(frame**2)
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, frames=range(0, 10), init_func=init, blit=True)
plt.show()
```

---

## 10. Advanced Plots

### Quiver Plot
Used for vector fields.
```python
X, Y = np.meshgrid(range(5), range(5))
U, V = np.cos(X), np.sin(Y)
plt.quiver(X, Y, U, V)
plt.title("Quiver Plot")
plt.show()
```

---

### Fill Between
Highlight areas on plots.
```python
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.fill_between(x, y, y + 0.5, color="lightblue")
plt.title("Fill Between")
plt.show()
```

---

## 11. Event Handling

### Capturing User Inputs
```python
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 4])

def on_click(event):
    print(f"Mouse clicked at: {event.xdata}, {event.ydata}")

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
```

---

## 12. Add Decorations (Ornaments)

### Text and Annotations
```python
plt.plot([1, 2, 3], [1, 4, 9])
plt.text(2, 4, "Highlight", fontsize=12, color='blue')
plt.annotate("Peak", xy=(2, 4), xytext=(1, 7), arrowprops=dict(facecolor='black', arrowstyle="->"))
plt.show()
```

---

## 13. Keyboard Shortcuts in Matplotlib

### Basic Shortcuts
- **"s"**: Save figure as a file.
- **"p"**: Pan axes.
- **"o"**: Zoom.
- **"Ctrl"-"+" / "-"**: Zoom in/out.

### Testing Shortcuts
```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 1])
plt.show()
# Try pressing 's', 'p', or 'o'
```

---

## 14. Summary

With tools like Matplotlib, Seaborn, pandas Plotting, and Bokeh, Python provides a powerful toolkit for **both static and interactive visualizations**, supporting everything from basic trends to highly customized plots. Experimenting with these techniques can enhance your storytelling through data.
