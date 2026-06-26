import sys; sys.path.insert(0,'.')
import joblib, numpy as np, pandas as pd
from src.features.extractor import extract_joint_features

df = pd.read_csv('datafor/ncbi5000_final.csv')
seqs = df['Enzyme'].tolist(); sms = df['Substrates'].tolist()
print("Extracting features...", flush=True)
feat = extract_joint_features(sms, seqs).astype(np.float32)
feat = np.concatenate([feat[:,960:], feat[:,:960]], axis=1)

def pad_feat(feat, target_dim):
    if feat.shape[1] >= target_dim: return feat[:, :target_dim]
    return np.pad(feat, ((0,0), (0, target_dim - feat.shape[1])), constant_values=0)

preds = {}
model_dims = {'baseline': 1984, 'condition': 1986, 'msa1d': 1990, 'msa2d': 1998}
for mt, dim in model_dims.items():
    X = pad_feat(feat, dim)
    for task in ['km', 'kcat']:
        if mt == 'baseline':
            path = 'enzyme-model/artifacts/baseline/%s/%s_predictor.joblib' % (task, task)
        else:
            path = 'enzyme-model/artifacts/%s/%s/%s_predictor.joblib' % (mt, task, mt)
        m = joblib.load(path)
        preds['%s_%s' % (mt, task)] = m.predict(X)
        print("%s/%s: [%.2f, %.2f]" % (mt, task, preds['%s_%s'%(mt,task)].min(), preds['%s_%s'%(mt,task)].max()), flush=True)

for task in ['km', 'kcat']:
    stack = joblib.load('enzyme-model/artifacts/stacking_v2/%s/model.joblib' % task)
    X_stack = np.column_stack([preds['baseline_'+task], preds['condition_'+task],
                               preds['msa1d_'+task], preds['msa2d_'+task]])
    df['stack_'+task] = stack['model'].predict(X_stack)
    print("Stack %s: [%.2f, %.2f]" % (task, df['stack_'+task].min(), df['stack_'+task].max()), flush=True)

df['stack_kcat_km'] = df['stack_kcat'] - df['stack_km']
df = df.sort_values('stack_kcat_km', ascending=False).reset_index(drop=True)
n = len(df); nt = int(n*0.10); cut = df['stack_kcat_km'].iloc[nt-1]
print('Total: %d  Top10%%: %d  cutoff=%.3f (kcat/Km=%.0f)' % (n, nt, cut, 10**cut), flush=True)

for ss in ['AcAP','TcAP','EaAP','KoAP','MnAP','MsAP']:
    m = df[df['Entry'].str.contains(ss, na=False, regex=False)]
    if len(m) > 0:
        r = int(m.index[0]) + 1
        v = float(m['stack_kcat_km'].iloc[0])
        print('  %s: #%d/%d  log10=%.2f  kcat/Km=%.0f  %s' % (ss, r, n, v, 10**v, 'GREEN' if r <= nt else 'GRAY'), flush=True)

df.to_csv('datafor/ncbi5000_stacking.csv', index=False)
print('DONE', flush=True)
