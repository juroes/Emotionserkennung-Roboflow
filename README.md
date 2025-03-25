# Emotion-Detection Roboflow Dataset

[face emotion Classification Dataset (v1, 2022-12-28 8:39pm) by Hung](https://universe.roboflow.com/hung-5yuey/face-emotion-8vfzj/dataset/1#)

## Klassifizierung auf Basis eines Roboflow-Dataset

Es soll ein Dataset von Roboflow genutzt und trainiert werden, um ein Modell für die Emotionserkennung zu erstellen, das anschließend in einem Anwendungsbeispiel verwendet wird.

### **Technologien**

- **TensorFlow**: Für die Erstellung und das Training des CNN-Modells.
- **Keras**: Hochwertige API für die Modelldefinition und das Training.
- **Roboflow**: Plattform zur Datenvorbereitung und -bereitstellung von vorab vorbereiteten Datasets.
- **H5**: Format für das Speichern und Weiterverwenden des Modells.

## Anwendung

Dazu habe ich einen Code geschrieben, der es ermöglicht, ein Bild hochzuladen und die Emotion darauf zu erkennen. Sobald ich ein Bild auswähle, verarbeitet das Programm es und zeigt die Vorhersage im Fenster an. Ich habe dafür Tkinter verwendet, um eine einfache Benutzeroberfläche zu erstellen. Ich habe es zuerst mit einer Webcam und Live Emotions Bestimmung versucht, was aber leider nicht funktioniert hat.

## **Fazit**

Es ist wichtig, den Datensatz korrekt vorzubereiten, geeignete Modelle zu wählen und die Hyperparameter anzupassen, um die Leistung zu steigern. Leider gab es Probleme bei der Modelloptimierung, was sich auf das Ergebnis auswirkte. Dennoch bietet der Ansatz eine solide Grundlage für emotionale Bildklassifikation und kann mit weiteren Anpassungen und Verbesserungen verbessert werden.

### Quellen

https://universe.roboflow.com/hung-5yuey/face-emotion-8vfzj/dataset/1#

https://www.youtube.com/watch?v=wuZtUMEiKWY

https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-classification-on-custom-dataset.ipynb
