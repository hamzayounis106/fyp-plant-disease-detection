# COLAB quickstart

Use `colab.ipynb` to run the full `RUNBOOK.md` pipeline on free Colab (T4) with Drive-backed persistence and `--resume`.

## Prereqs (one-time)

- Create `MyDrive/TRAINED/`.
- Put Kaggle API token at `MyDrive/TRAINED/kaggle.json` (Kaggle -> Account -> Create New API Token).
- Clone this repo once to `MyDrive/TRAINED/code/`.

## Anti-idle (DevTools Console)

Paste this in browser DevTools while Colab tab is open:

```javascript
function KeepAlive(){
  document.querySelector("#top-toolbar > colab-connect-button")
          .shadowRoot.querySelector("#connect").click();
}
setInterval(KeepAlive, 60000);
```

If selector changes after a Colab UI update, keep the notebook tab focused manually.

## Time budget (free T4)

- End-to-end retrain + eval: about 7-10 hours GPU time.
- Typical split: setup/data 30-60 min, CNN 2-3 h, EfficientNet 3.5-5 h, eval 10-20 min.
- Expect quota limits; spread long runs across 2-3 sessions.

## Recovery procedure

If disconnected mid-training:

1. Reconnect and rerun cells 1, 2, 3, 4, 10, 11.
2. Rerun interrupted training cell (12 or 13).

`train.py --resume` continues from `last.pt` synced in Drive.
