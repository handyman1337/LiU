Commandon att köra innan start:

    pip install torch
    pip install open-clip-torch

    ladda ned "RSICD_images" och "txtclasses_rsicd" från lisam projektmapp


PROJEKT FÖRKLARKING:

    mappen "processed_data" innehåller 3 json filer som är splittrade i dataA, dataB och dataC och innehåller ImageFileName, Tokens, Labels (viktigaste). 

    mappen "RSICD_images" innehåller alla faktiska jpg image filer.

    filen "split_and_label.py" är den som skapar dessa och i princip kombinerar "txtclasses_rsicd" och "dataset_rsicd.json" ihop samtidigt som den splitrar till A,B,C dataset.

    Så när det gäller dataset så bör endast "processed_data" och "RSICD_images" vara viktigt vi behöver inte kolla mer på "dataset_rsicd.json" och "txtclasses_rsicd".



Validation hur funkar (för grade 3)

    Vår modell kommer tidigare ha tränats så vi har en image embedder och en text embedder där en image med en caption ska ge så nära embedding som möjligt till varandra.

    För zero-shot classification så skippar vi helt i captionen för dataC och har istället 20+ classes som ser ut följande: "A satellite image of airport", "A satellite image of forest", "A satellite image of river"  

    Dessa texter körs genom vår tränade text embedder och sparas.

    För validation så tar den ett image från dataC genom image embedder och sedan tittar vilken av de här class text embeddersarna är mest likt vårt resultat, den väljer en och jämför med label.