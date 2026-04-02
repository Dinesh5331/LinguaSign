import cv2
import torch
import numpy as np
import pickle
from collections import deque
import HandTrackingModule as htm

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn as nn

# ===================== LOAD LABEL ENCODER =====================
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ===================== DEFINE MODEL =====================
class GNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = GCNConv(2, 64)
        self.bn1 = BatchNorm(64)

        self.conv2 = GCNConv(64, 128)
        self.bn2 = BatchNorm(128)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.fc(x)

# ===================== LOAD MODEL =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GNNModel(len(le.classes_)).to(device)
model.load_state_dict(torch.load("gnn_model.pth", map_location=device))
model.eval()

# ===================== HAND GRAPH EDGES =====================
edges = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

# ===================== SMOOTHING =====================
pred_queue = deque(maxlen=5)

# ===================== LOOP =====================
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, Draw=False)

    if len(lmList) == 21:
        xs = [lm[1] for lm in lmList]
        ys = [lm[2] for lm in lmList]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        if width != 0 and height != 0:
            wrist_x = lmList[0][1]
            wrist_y = lmList[0][2]

            coords = []
            for lm in lmList:
                norm_x = (lm[1] - wrist_x) / width
                norm_y = (lm[2] - wrist_y) / height
                coords.append([norm_x, norm_y])

            x = torch.tensor(coords, dtype=torch.float)

            graph = Data(x=x, edge_index=edge_index)
            graph.batch = torch.zeros(x.shape[0], dtype=torch.long)

            graph = graph.to(device)

            with torch.no_grad():
                out = model(graph)
                prob = torch.softmax(out, dim=1)

                confidence, pred = torch.max(prob, dim=1)

            label = le.inverse_transform([pred.item()])[0]
            conf = confidence.item()

            # ===================== SMOOTHING =====================
            pred_queue.append(label)
            final_label = max(set(pred_queue), key=pred_queue.count)

            # ===================== DISPLAY =====================
            cv2.putText(img,
                        f"{final_label} ({conf:.2f})",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3)

    cv2.imshow("GNN Sign Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()