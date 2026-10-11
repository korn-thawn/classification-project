export default function ResultCard({ result }) {
  return (
    <div className={`result ${result.result}`}>
      <div className="verdict">{result.result}</div>
      <dl>
        <dt>score (P(NG))</dt><dd>{result.score.toFixed(4)}</dd>
        <dt>threshold</dt><dd>{result.threshold.toFixed(4)}</dd>
        <dt>model</dt><dd>{result.model_version}</dd>
        <dt>inference</dt><dd>{result.latency_ms} ms</dd>
      </dl>
    </div>
  );
}
