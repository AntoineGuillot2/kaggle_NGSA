# Kaggle NGSA
Avant de lancer les scripts, donwloads les data depuis https://drive.google.com/file/d/1WNwWtmFg6kuyFektO5CfvjuleBDWAq_d/view?usp=sharing puis extraire dans /data.

Pipeline:
- On change l'id de l'article par un id entre 0 et 27700 (nombre d'articles)
- Transformation du graph en embedding avec LINE (1 Matrices N par D)
- Transformation des abstract et des titles avec Doc2Vect (1 Matrices N par K1, 1 Matrice N par K2)
- Concatenation des 3 matrices, on ajoute aussi l'année de publication des articles
- On normalise column-wise
- Train du FCN (256-512) avec dropout
- Prévision sur le test set
