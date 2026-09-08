# it's giving

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/aa5ed48f-70c2-4022-ac7f-87a4c3066a24" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/c764c5eb-c17a-47f2-b4b0-49153c8cb3c0" width="100%"></td>
  </tr>
</table>

Pull a face at your webcam. It works out *which* face, and drops the matching
meme over your head, scaled to follow you around the frame. You can extend and
add more memes to your heart's desire.

Point Zoom at its virtual camera and the whole call sees it.

```bash
python its_giving.py              # preview + virtual camera
python its_giving.py --no-vcam    # preview only
```

Fourteen reactions: time out, heart hands, hands over face, crashing out,
dancing, nose pinch, flirty, hand up, tongue out, gasp, disgust, talking to the
wall, side-eye, and spinning.

The whole thing is one file. There is no config, no calibration step, and no
build — you run it and it works, and when it doesn't you open `its_giving.py`
and change a number.

---

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Three MediaPipe models (~15 MB) download themselves the first time you run it,
into `models/`.

---

## Start here

```bash
python its_giving.py
```

The HUD is on by default, and it's the whole tuning story. You get a box around
your face, dots on your palms and elbows, and a live readout of every number the
detector is looking at:

```
showing: -   raw: -   hands: 0   elbows up: n
jaw 0.03  tongue 0.00  turn 0.04  squint 0.11  gesture 0.002
disgust 0.18 = 2x sneer 0.02 + brow 0.09 + frown 0.03 + lip 0.02
```

`raw` is what the frame matched this instant. `showing` is what actually made it
past the arm/hold timers onto your face. When something feels wrong, the gap
between those two columns tells you which half to fix.

---

## Using it in meetings

The virtual camera is on by default. Zoom, Meet, Teams, Discord and OBS all
treat it as a normal webcam.

**1. Install a backend** (once):

| OS | do this |
|---|---|
| macOS | install [OBS Studio](https://obsproject.com), open it once, quit it |
| Windows | install OBS Studio, or run its virtual-camera installer |
| Linux | `sudo apt install v4l2loopback-dkms` then `sudo modprobe v4l2loopback` |

**2. Run it.** It prints the device name it's publishing to:

```
Virtual camera: 'OBS Virtual Camera'  <- pick this camera in Zoom / Meet
```

If the backend isn't installed it says so and carries on in preview-only mode
rather than dying.

**3. Pick that device in your meeting app:**

- **Zoom** — Settings → Video → Camera → *OBS Virtual Camera*
- **Google Meet** — ⚙ Settings → Video → Camera → *OBS Virtual Camera*
- **Teams** — ⚙ Settings → Devices → Camera → *OBS Virtual Camera*
- **Discord** — Settings → Voice & Video → Video Device

**Start this *before* your meeting app.** Most of them scan for cameras once at
launch and won't notice a device that appeared later. If it's not in the list,
quit the meeting app and reopen it.

### Before you turn it on in a real meeting

It fires on its own — everyone sees whatever it decides. Try it on a call with
someone who likes you before a call with your CEO.

Reactions land over your face in *your* video tile, not on the shared screen.
`talking_to_wall`, `suspicious` and `dance` are the three most likely to go off
during completely normal meeting behaviour — gesturing while you talk,
concentrating, and leaning back — so if you're nervous, raise their `ARM` counts
or take them out of `POSES` for the day.

Two gotchas: the preview window has to be focused for `q` to work, and quitting
mid-call leaves the virtual camera showing nothing, so switch your camera back
in the meeting app *first*.

---

## How it works

```
camera frame
     |
 1.  MediaPipe    face: 478 landmarks + 52 blendshapes
                  hands: 2 x 21 points
                  body: shoulders, elbows, wrists
     |
 2.  Measures     face-relative distances, tongue colour, hand speed
     |
 3.  decide()     one ordered pass -- first pose that matches wins
     |
 4.  arm / hold   must persist N frames to fire, lingers 10 frames after
     |
  overlay         asset scaled to your face, alpha-composited, GIFs animated
```

### Everything is measured in face widths

Pixel coordinates depend on how far you're sitting from the lens, so nothing is
compared in pixels. Distances are divided by the width of your face box first —
`near(hand.index, face.mouth, 0.22)` means "within 22% of a face width", and
that means the same thing at 40 cm and at a metre and a half. Hand speed gets
the same treatment: face-widths per frame.

Head turn is the nose's position between the two edges of your face, `0` facing
the camera, about `0.4` in full profile. That one's free — no distance
normalisation needed at all.

### First match wins

There's no scoring, no arbitration, no competition between poses. `decide()`
walks the list top to bottom and returns the first thing that fits, so the
order in `POSES` *is* the priority order. Two-hand poses come first (a time-out
T and heart hands are specific and hard to trigger by accident), then
one-hand poses, then face-only expressions, then the vague ambient ones —
`talking_to_wall` and `suspicious` sit near the bottom for exactly that reason.

The look-alike problem gets solved by ordering plus a few explicit exclusions
rather than by a suppression graph. `flirty` is a fingertip on your lips *and*
your palm not covering your mouth, which is what keeps it from stealing
`cover_nose`. `dance` and `crashing_out` are the same body pose — elbows above
your shoulders — split apart by whether your mouth is open.

### The two measurements that aren't landmarks

**Tongue** is a colour test, not a landmark. MediaPipe has no tongue. So when
your jaw is already open past 0.3, the inner-lip ring is filled into a polygon,
eroded 15% inward so no lip pixels sneak in, and the fraction of pixels inside
that are pink-and-saturated-and-lit gets measured in HSV. Teeth are
unsaturated; the back of your throat is dark; a tongue is neither. If a hand is
anywhere near your mouth the test is skipped entirely, because skin is pink too.

**Gesturing** is smoothed hand speed. Each frame, every palm is matched to its
nearest palm from the previous frame; jumps of more than a face width are
thrown away as re-detections rather than movement. The result runs through an
exponential smoother at 0.8, so `talking_to_wall` needs sustained waving rather
than one quick reach for your coffee.

### Two timers instead of a state machine

**Arm** — a pose has to hold for N consecutive frames before it fires. Default
is 3 (about a tenth of a second), and the noisy ones cost more: `spin` needs 15,
`suspicious` 8, `talking_to_wall` and `dance` 6. This is what kills the
single-frame flickers.

**Hold** — once a reaction fires it stays for 10 more frames after the pose
stops matching, so a momentarily lost landmark doesn't make the overlay strobe.

Your face box is smoothed too, at 0.7/0.3, and when the face disappears the
overlay stays exactly where it was rather than snapping to a corner. That's what
makes `spin` work: your face going missing *is* the trigger, so the meme has to
stay put once it's gone.

### The preflight check

Before anything else, the script opens a face detector in a throwaway
subprocess. A bad macOS MediaPipe build kills the process with `abort()`, which
no `try/except` can catch — so it's better to spend that crash somewhere it
doesn't matter and print instructions. `--skip-check` skips it once you know
your install is fine; it costs about a second at startup.

---

## The reactions

| pose | do this | shows |
|---|---|---|
| `time_out` | referee's T — one hand flat on top, one vertical underneath | `time_out.jpeg` |
| `heart` | two hands, index tips together, thumb tips together | `heart.jpeg` |
| `cover_nose` | both hands over your nose and mouth | `cover_nose.jpeg` |
| `crashing_out` | both hands to your head, mouth open | `crashing_out.jpeg` |
| `dance` | both hands up behind your head, mouth closed | `dance.jpeg` |
| `nose_closed` | pinch your nose shut | `nose_closed.gif` |
| `flirty` | one index fingertip on your lips | `flirty.jpeg` |
| `hand_up` | one open palm up beside your head | `hand_up.jpeg` |
| `tongue_out` | tongue out, mouth open | `tongue_out.jpeg` |
| `open_mouth` | jaw drops | `open_mouth.jpeg` |
| `disgusted` | scrunch your nose, or brows down and frown | `disgusted.jpeg` |
| `talking_to_wall` | hands in frame, gesturing away | `talking_to_wall.gif` |
| `suspicious` | turn your head and squint | `suspicious.jpeg` |
| `spin` | leave the frame entirely | `spin.gif` |

**Three of them are worth knowing about up front.**

`spin` fires when your face, your hands *and* your body have all left the frame
— it's "you're gone", not "you turned around". Fifteen frames of nothing, so
walking past the camera won't set it off.

`dance` and `crashing_out` share one detector. Elbows above shoulder height with
your hands near your head is the pose; the only thing separating them is
`jawOpen` against `scream_jaw` (0.3). Silent flailing gets you `dance`.

`disgusted` fires on a nose scrunch alone (`noseSneer` past 0.12 — a resting
face reads about 0.02) *or* on the weighted sum of four channels
`2×sneer + browDown + mouthFrown + upperLipRaise` past 0.6. Any one of those
four alone is too weak to trust, but a real scrunch moves all of them together.

---

## Getting good results

**Light your face from the front.** A window behind you is the single biggest
cause of missed detections — the landmarks go unreliable and everything gets
harder to trigger.

**Get reasonably close.** Head and shoulders. The tongue test in particular
needs a mouth opening of at least 8×8 pixels with 40 usable pixels inside it,
which you will not have from across the room.

**Keep your hands in frame** for the hand poses, and your shoulders in frame for
`dance` and `crashing_out` — those read your elbows, not your hands, precisely
because the hand model loses your hands once they're behind your head.

**Commit for a beat.** The arm counters are deliberate. A pose pulled and
instantly dropped is indistinguishable from a bad frame.

**Drop the capture size if it's choppy.** Three MediaPipe models run on every
frame. `--size 640x480` is the fix.

---

## Tuning it

Everything lives in two blocks at the top of `its_giving.py`.

```python
T = dict(
    jaw_open=0.5,     # open_mouth
    scream_jaw=0.3,   # mouth open this much + elbows up = crashing_out, below = dance
    tongue_jaw=0.3,   # jaw must be this open before the tongue test even runs
    tongue=0.5,       # fraction of the mouth opening that reads pink
    sneer=0.12,       # nose scrunch on its own
    disgust=0.6,      # ...or the four-channel sum
    head_turn=0.15,   # 0 = facing camera, ~0.4 = full profile
    squint=0.3,
    gesture=0.035,    # hand speed, face-widths per frame
)

ARM = {"spin": 15, "suspicious": 8, "talking_to_wall": 6, "dance": 6,
       "crashing_out": 4, "open_mouth": 4, "tongue_out": 5, "disgusted": 5}
```

Turn the HUD on, do the thing, read the number off the screen, and set the
threshold between what you get when you mean it and what you get when you don't.
That second number is the one people skip, and it's the one that decides whether
this is fun or annoying.

| what's happening | what to change |
|---|---|
| fires when you shift in your seat | raise its `ARM` count first — it costs you almost nothing |
| fires, but the *wrong* pose | move it earlier in `POSES`, or tighten the one that's stealing it |
| never fires | HUD: is the number moving at all? Flat means detection (light, framing, distance). Moving but under the line means threshold |
| flickers on and off | raise `HOLD_FRAMES` (default 10) |
| meme is too big for your video tile | lower `FACE_SCALE` (default 2.0 — try 1.6) |
| everything is too eager | nudge the whole `T` block up 10% |

Two more constants worth knowing: `FACE_SCALE = 2.0` sets the overlay height as
a multiple of your face height, and `HOLD_FRAMES = 10` is the linger. To turn a
pose off entirely, delete it from `POSES` — but keep `TEST_KEYS` the same length
or the keys shift under you.

---

## Adding your own reaction

### Just swapping the image — 30 seconds

Drop a file in `assets/` named after the pose: `heart.png`, `heart.gif`,
`heart.jpg`, `heart.jpeg`. A `something_` prefix is ignored, so
`123_heart.jpeg` works too, which is handy when you're saving straight out of a
browser.

JPEG, PNG and animated GIF all work. Transparency is respected — a PNG or GIF
with an alpha channel composites over your video properly instead of arriving in
a black box. GIF frame timings are read from the file, so animations play at
their real speed, looping from the moment the reaction fired. The image is
scaled to `FACE_SCALE` times your face height and centred on your head, so its
aspect ratio survives whatever shape it is.

A missing asset isn't fatal: you get a red circle with the pose name on it, and
a line at startup telling you which ones are missing.

### A whole new pose — three steps

**1. Add it to `POSES`**, at the priority you want. Earlier is stronger.

```python
POSES = ["time_out", "heart", "thinking", "cover_nose", ...]
```

**2. Add a branch to `decide()`.** You have `face`, `hands`, `body`, `tongue`
and `gesture`, plus `near(a, b, k)` for "within k face widths":

```python
    for h in hands:
        if near(h.palm, face.chin, 0.5) and not h.open and jaw < 0.2:
            return "thinking", d
```

What you have to work with:

| | |
|---|---|
| `face.b("jawOpen")` | any of the 52 blendshapes, 0–1 |
| `face.nose`, `face.chin`, `face.top`, `face.mouth`, `face.center` | anchor points, in pixels |
| `face.w`, `face.h`, `face.box`, `face.eye_y` | face geometry |
| `face.turn` | head turn, 0 facing camera to ~0.4 in profile |
| `face.pts[i]` | any of the 478 raw landmarks |
| `h.palm`, `h.thumb`, `h.index`, `h.middle` | hand points |
| `h.open`, `h.horizontal`, `h.vertical` | hand shape |
| `body.elbows_up`, `body.shoulders`, `body.elbows`, `body.wrists`, `body.seen` | upper body |
| `near(a, b, k)` | distance test in face widths |
| `tongue`, `gesture` | the two computed measures |

**3. Add it to the HUD** if it's driven by a new number — put it in the `d`
dict in `decide()` and print it in `draw_hud()`. You will not be able to tune it
otherwise, and tuning blind is how you end up with a pose that fires at your
neighbour's cat.

Then test it properly: do the pose and watch it fire, then do everything
*adjacent* to it — the poses it might get confused with — and watch it stay
quiet. Set your threshold above that second number, not just below the first.

Drop your asset in `assets/` under the same name and you're done.

---

## Options

```bash
python its_giving.py --camera 1          # pick a webcam (0 is often your iPhone)
python its_giving.py --size 640x480      # smaller capture, faster
python its_giving.py --no-flip           # don't mirror
python its_giving.py --no-vcam           # preview only
python its_giving.py --skip-check        # skip the MediaPipe preflight
```

---

## When something's wrong

| problem | what's going on |
|---|---|
| `Check failed: service_ Service is unavailable` | MediaPipe 0.10.30+ on Apple Silicon. Reinstall 0.10.21 — the script prints the exact commands |
| exits with a MediaPipe message before the camera opens | that's the preflight doing its job; follow what it printed |
| camera won't open | try `--camera 1`. On macOS your *terminal* needs camera permission: System Settings → Privacy & Security → Camera |
| nothing ever fires | watch the HUD. Flat numbers mean detection — light, framing, distance. Moving numbers that stay under the line mean thresholds |
| fires constantly | raise the `ARM` count for that pose first, then its threshold in `T` |
| wrong reaction shows up | move yours earlier in `POSES`, or tighten whatever is stealing it |
| reaction flickers | raise `HOLD_FRAMES` |
| slow / choppy | `--size 640x480` |
| hand poses never work | your face *and* hands both need to be in frame — hands are measured against your face |
| `dance` / `crashing_out` never work | your shoulders need to be in frame; those read elbows, not hands |
| red circle with a name on it | no asset for that pose. Drop one in `assets/` |
| meeting app can't see the camera | start this first, then the meeting app |
| no virtual camera at all | install OBS Studio (macOS/Windows) or `modprobe v4l2loopback` (Linux) |

---

## What's where

```
its_giving.py      everything: models, assets, detection, overlay, virtual camera
requirements.txt   see the note about the MediaPipe version
reaction_cam.py    an earlier copy of the script, kept around
assets/            the memes, named after their pose
models/            MediaPipe .task files (downloaded on first run, gitignored)
```

Inside `its_giving.py`, roughly in order:

```
POSES, T, ARM      the tuning block                    <- start here
ensure_models()    downloads the three .task files
preflight()        catches broken MediaPipe builds in a subprocess
Asset, load_asset  images and GIFs, scaling, frame timing
overlay()          alpha compositing
Face, Hand, Body   landmarks -> face-relative geometry
tongue_score()     HSV pink fraction inside the inner lips
Motion             smoothed hand speed
decide()           the ordered pose checks              <- and here
draw_hud()         the debug readout
main()             camera loop, arm/hold, test keys, virtual camera
```

Want to change **what sets off what**? `decide()`.
Want to change **how easily it goes off**? `T` and `ARM`.
