export function TrainingPage() {
  const cmd = `python train.py --dataset data/kcat-over-Km-data_0.4simi-10fold.csv --epochs 500 --test-every 5 --test-patience 10 --lr-patience 2 --lr 5e-4 --hidden-dim 192 --dropout 0.45 --weight-decay 0.03 --train-noise-std 0.01`;
  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="mb-6 text-2xl font-bold text-gray-900">Model Training (CLI)</h2>
      <div className="rounded-xl border border-gray-200 bg-white p-6">
        <p className="mb-3 text-sm text-gray-500">
          Training runs 10-fold CV over ~25,000 samples. Recommended on Linux with GPU.
        </p>
        <div className="rounded-lg bg-gray-900 p-4 font-mono text-sm text-green-300 overflow-auto">
          {cmd}
        </div>
      </div>
    </div>
  );
}
