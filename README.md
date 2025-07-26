# Adaptation of a Robotic Hand Prototype in Hardware and Software for Grasp Emulation Using Machine Learning Methods
Author: Pablo Kevin Pereira\
Email: pablo-kevin@ieee.org\
UTEC - Universidad Tecnológica del Uruguay, ITR Suroeste

Tutor: José Chelotti\
Email: jchelotti@sinc.unl.edu.ar\
UNL - Universidad Nacional del Litoral, Argentina\

Co-Tutor: Lucas Baldezzari\
Email: lucas.baldezzari@utec.edu.uy\
UTEC - Universidad Tecnológica del Uruguay, ITR Suroeste

## Abstract
This project focused on the adaptation of a robotic hand prototype in both hardware and software, with the goal of emulating tool grasps using Machine Learning methods. The developed solution was a mechatronic system composed of four main modules.
- Module 0: A machine-to-machine control interface between the system modules, along with a command-line user interface.
- Module 1: Implementation of an image segmentation pipeline (Segmentator), a convolutional neural network for image classification (Observer), and a deep reinforcement learning agent for predicting tool grasps (Actor).
- Module 2: Development of a simulation of the robotic hand prototype using ROS2.
- Module 3: Development of the control firmware and electronic circuit for the robotic hand prototype; control was divided between MQTT-based communication and servo motors control, which transmitted movement to the robotic fingers through cables emulating elastic tendons.
- An additional module focused on the verification of the others, especially performance measurements, results visualization, and brief analysis and hypotheses regarding the neural networks' learning behavior.

The datasets were generated through data augmentation techniques and included nine types of tools and an empty image (to teach the agent to fully open the hand when no object is present). The ten classes were: empty, nut, screw, nail, pen, fork, spoon, screwdriver, hammer, and pliers. The Segmentator was based on a pretrained model and took an RGB image as input, returning a black-and-white masked image. The Observer, trained via supervised learning, used this mask as input and output logits, which were used to classify the tools with 75% precision.

These logits fed the DRL agent implemented with the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, which consisted of six deep neural networks trained in a simulation environment also developed by the author. The reward function was designed to promote finger combinations coherent with functional grasps. The core of TD3 was the Actor, which took the logits and finger index as input and output the closing level for each finger. The Actor reached 88% precision in predicting appropriate grasps during the verification phase. This verification started from the Segmentator, passed through the Observer, and concluded with the Actor’s output.

The results demonstrated the potential of this architecture to generalize grasps beyond a fixed set of tools.

This work has contributed a functional intelligent grasping system and stands out for its effective integration of computer vision, deep learning, and embedded systems.


>**Keywords — _Robotic Hand, Grasp Emulation, Deep Reinforcement Learning, TD3, Computer Vision, ESP32, ROS2._**


## Resumen
Este proyecto desarrolló un sistema mecatrónico compuesto por cuatro módulos principales.
- Módulo 0: una interfaz de control Machine-to-Machine entre los componentes del sistema, que incluye una interfaz de usuario por línea de comandos.\
- Módulo 1: implementación de un pipeline de segmentación de imágenes (Segmentator), una red neuronal convolucional para clasificación de herramientas (Observer) y un agente de aprendizaje por refuerzo profundo para la predicción de agarres (Actor).
- Módulo 2: simulación del prototipo de mano robótica utilizando ROS2.
- Módulo 3: desarrollo del firmware de control y del circuito electrónico del prototipo físico; el control se dividió entre comunicación MQTT y control de servomotores, que transmiten el movimiento a los dedos robóticos mediante cables que emulan tendones elásticos. 
- Se incorporó un módulo adicional de verificación de los anteriores, centrado en mediciones de rendimiento, visualización de resultados y un breve análisis e hipótesis sobre el aprendizaje de las redes neuronales. 

Los datasets fueron generados mediante técnicas de aumentación de datos e incluyeron nueve tipos de herramientas y una imagen en blanco. El Observer, entrenado con aprendizaje supervisado, produjo salidas que permitieron clasificar las herramientas con un 75% de precisión. Estas salidas alimentaron al agente de DRL, implementado con el algoritmo Twin Delayed Deep Deterministic Policy Gradient (TD3), compuesto por seis redes neuronales profundas entrenadas en un entorno simulado desarrollado en este proyecto. La función de recompensas fue diseñada para promover combinaciones de dedos coherentes con agarres funcionales. El Actor alcanzó un 88% de precisión en la predicción de agarres adecuados.

>**Palabras claves — _Mano Robótica; Emulación de Agarres; Aprendizaje por Refuerzo Profundo; TD3; Visión por Computadora; ESP32; ROS2._**