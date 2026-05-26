from thermography import detectEdges
from tkinter import filedialog

if __name__ == '__main__':
    frame = "E:/410SS DATA/modified datasets/L wall 400C Interpass 121225/data_collection_20251212_141949/FLIR/FLIR-Frame-580.npy"

    detectEdges(frame, 25, 40)