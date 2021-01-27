import numpy as np

def sim(X,y,model,PRELEV, MAX_BET):
    cost = [0]
    curr = [0]
    rend = []
    races= []
    wins = []
    ratio = []
    vals = []
    bank = [0]
    days = pd.to_datetime(X['jour'].unique()).sort_values()

    for day in tqdm(days):
        lines = X.jour==day
        X_day = X[lines]

        y_pred_prob = model.predict_proba(X_day)
        y_pred = np.argmax(y_pred_prob, axis=1)

        y_day = y[lines]

        N = len(y_day)
        b = bank[-1]

        if curr[-1] < N:
            missing = N - curr[-1]
            from_bank = min(missing,b)
            b -= from_bank
            missing -= from_bank

            cost.append(cost[-1]+missing)
            val = 1
        else:
            cost.append(cost[-1])
            val = min(round(curr[-1]/N,2),MAX_BET*curr[-1])

        vals.append(val)
        c = curr[-1] - val * N
        cote = []
        for pred, cl, (i,x) in zip(y_pred,y_day, X_day.iterrows()):
            if pred == cl:
                if pred == 1:
                    cote.append(x['dernier_rapport_direct'])
                else:
                    cote.append(x[f'dernier_rapport_direct_{pred}'])
        ratio.append(round(len(cote)/N,2))
        rend.append(sum(cote)/N)

        c += sum(cote)*val
        c = max(0,c)
        bank.append(b+c*PRELEV)
        curr.append(c*(1-PRELEV))
        races.append(N)
        wins.append(len(cote))

    return {
        'days':days,
        'bank':bank[1:],
        'cost':cost[1:],
        'curr':curr[1:],
        'vals':vals,
        'rend':rend,
        'races':races,
        'wins':wins,
        'ratio':ratio
    }