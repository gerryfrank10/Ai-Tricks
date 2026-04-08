# AI in Automation & Robotics

AI is giving robots the ability to learn, adapt, and understand natural language instructions. In 2025, the frontier is embodied AI — models that can perceive, reason, and act in the physical world.

---

## 2025 Landscape

| Capability | State of the Art | Key Tools |
|-----------|-----------------|---------|
| Object Detection | YOLOv11, RT-DETR | Ultralytics, Hugging Face |
| Grasping | Diffusion Policy, ACT | LeRobot |
| Task Planning | GPT-4o, Claude + Code | LangChain, custom agents |
| Navigation | Nav2, RL-based | ROS2, Isaac Sim |
| Foundation Models | RT-2, π0, GROOT | Google DeepMind, Physical Intelligence |

---

## Computer Vision for Robotics — YOLO Object Detection

```python
from ultralytics import YOLO
import cv2, numpy as np

# Real-time detection from camera
model = YOLO("yolo11n.pt")   # Nano for real-time on edge hardware

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.5,
        classes=[39, 41, 64, 67],    # bottle, cup, scissors, phone
        device="cuda",
        verbose=False,
    )

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = model.names[int(box.cls)]
            conf     = float(box.conf)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Compute centroid for robot targeting
            cx, cy = (x1+x2)//2, (y1+y2)//2
            print(f"Target: {cls_name} at pixel ({cx}, {cy})")

    cv2.imshow("Robot Vision", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Imitation Learning — Learning from Human Demonstrations

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DemonstrationDataset(Dataset):
    """Dataset of human demonstrations: (observation, action) pairs."""

    def __init__(self, obs_path: str, actions_path: str):
        self.observations = np.load(obs_path)    # (N, obs_dim)
        self.actions      = np.load(actions_path) # (N, action_dim)

    def __len__(self): return len(self.observations)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.observations[idx]),
            torch.FloatTensor(self.actions[idx])
        )

class BehaviorCloner(nn.Module):
    """Behavioral Cloning: supervised learning on demonstrations."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

# Training
dataset = DemonstrationDataset("obs.npy", "actions.npy")
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

model = BehaviorCloner(obs_dim=64, action_dim=7)   # 7-DoF robot arm
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    total_loss = 0
    for obs, actions in loader:
        pred = model(obs)
        loss = nn.MSELoss()(pred, actions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(loader):.6f}")
```

---

## LeRobot — Hugging Face Robotics Library

```python
# pip install lerobot
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load pre-collected demonstration dataset
dataset = LeRobotDataset("lerobot/pusht")

# ACT Policy (Action Chunking with Transformers)
policy = ACTPolicy.from_pretrained("lerobot/act_pusht")
policy.eval()

# Inference — given observation, predict action chunk
import torch
obs = {
    "observation.state": torch.randn(1, 2),           # robot state
    "observation.image": torch.randn(1, 3, 96, 96),   # camera image
}

with torch.no_grad():
    action = policy.select_action(obs)
print(f"Predicted action: {action}")
```

---

## LLM Task Planning for Robots

```python
import anthropic

client = anthropic.Anthropic()

# Available robot primitives
ROBOT_TOOLS = [
    {
        "name": "move_to",
        "description": "Move robot end-effector to a named location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Named location: 'table', 'bin', 'shelf_A', 'charger'"},
                "speed": {"type": "string", "enum": ["slow", "normal", "fast"], "default": "normal"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "grasp_object",
        "description": "Grasp an object at the current location",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "grip_force": {"type": "number", "description": "Grip force 0-100", "default": 50}
            },
            "required": ["object_name"]
        }
    },
    {
        "name": "release_object",
        "description": "Release the currently held object",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "get_scene_description",
        "description": "Get a description of objects visible in the robot's camera",
        "input_schema": {"type": "object", "properties": {}}
    }
]

def execute_robot_task(natural_language_command: str) -> str:
    """Use Claude to plan and execute a robot task."""
    messages = [{
        "role": "user",
        "content": f"Robot task: {natural_language_command}"
    }]

    system = """You are a robot task planner. Use the available tools to execute the given task.
Think step by step. Observe the scene first if you need to identify objects.
Execute one action at a time and check results before proceeding."""

    for _ in range(20):   # Max 20 steps
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            tools=ROBOT_TOOLS,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return response.content[-1].text

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_robot_primitive(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "user", "content": tool_results})

    return "Task execution limit reached"

# Example
result = execute_robot_task("Pick up the red bottle from the table and place it in bin_A")
print(result)
```

---

## ROS2 + ML Integration

```python
# ROS2 node that runs YOLO detection on camera stream
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector")
        self.model = YOLO("yolo11n.pt")
        self.bridge = CvBridge()

        # Subscribe to camera
        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )
        # Publish detections
        self.pub = self.create_publisher(Detection2DArray, "/detections", 10)

        self.get_logger().info("YOLO Detector Node started")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Run detection
        results = self.model.predict(cv_image, conf=0.5, verbose=False)

        # Publish detections
        det_array = Detection2DArray()
        det_array.header = msg.header

        for r in results:
            for box in r.boxes:
                det = Detection2D()
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                det.bbox.center.position.x = (x1 + x2) / 2
                det.bbox.center.position.y = (y1 + y2) / 2
                det.bbox.size_x = x2 - x1
                det.bbox.size_y = y2 - y1
                det_array.detections.append(det)

        self.pub.publish(det_array)

def main():
    rclpy.init()
    node = YOLODetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

---

## Tips & Tricks

| Challenge | Approach |
|-----------|---------|
| Sim-to-real gap | Domain randomization + real data fine-tuning |
| Limited demonstrations | Data augmentation, pre-trained vision encoders |
| Safety constraints | Hard-coded joint limits + LLM guardrails |
| Real-time requirements | TensorRT export, edge GPU (Jetson Orin) |
| Task generalization | Foundation model + in-context learning |

---

## Related Topics

- [Reinforcement Learning](../specialized/reinforcement-learning.md)
- [Computer Vision](../computer-vision/index.md)
- [AI Agents](../llm/agents.md)
