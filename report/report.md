## **Extract feature**  

```py
python3 feature.py --dataset dataset --features_db output/features.csv
```  

## **Cluster**  

```py
python3 cluster.py --features_db output/features.csv --clusters output/clusters.pickle
```

## **Build Bag of Visual Words**  

```py
python3 bovw.py --features_db output/features.csv --clusters output/clusters.pickle --bovw output/bovw.csv
```  

## **Construct Inverted Index**  

```py
python3 index.py --clusters output/clusters.pickle --bovw output/bovw.csv --index output/index.pickle
```  

## **Calculating TF-IDF**  

```py
python3 vect.py --bovw output/bovw.csv --index output/index.pickle --vector output/weight.csv
```  

## **Search an image**  

```py
python3 search.py --weights output/weight.csv --clusters output/clusters.pickle --dataset dataset --query queries/115100.png
```