# Beat the Agent

A 90-second phone game about reinforcement learning. The player runs two batteries on a
solar-heavy six-bus feeder for one day, then a tabular Q-learning agent learns the same task
in the browser (40,000 practice days) and plays the player's day.

- `index.html` – the game (no dependencies, no network calls, nothing recorded)
- `core.js` – feeder power flow, one-day environment, Q-learning (also runs in node)
- `test_core.js` – `node test_core.js` prints baseline, rule and trained-agent scores

Feeder, prices and penalties are illustrative.

Test hooks: `?auto=1` plays a fixed rule for the human (`?auto=idle` does nothing), `&at=K` jumps to step K, `?demo=result|cost|volt|reward|about`
jumps to a screen, `?step=120` changes the speed of the player's day (ms per 15 minutes).

## Try it on a phone on the same Wi-Fi

    cd rl_game && python3 -m http.server 8765
    # then open http://<this-computer's-IP>:8765 on the phone

## Publish with GitHub Pages

1. Create a public repository on github.com, for example `beat-the-agent`.
2. Upload `index.html` and `core.js` (Add file → Upload files), commit.
3. Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
4. The game is then at `https://<username>.github.io/beat-the-agent/`.
