# Vehicle Panel Damage — Annotation Guide

## Background

Images come from fixed cameras mounted in a vehicle scanning tunnel. Each image captures a vehicle panel (door, bonnet, bumper, quarter panel, etc.) as the vehicle passes through. Images have already been pre-annotated by an AI model — your job is to review, correct, and complete those annotations before they are used for training.

---

## Classes

There are exactly two classes:

| Class | Use it when… |
|-------|-------------|
| `scratch` | The paint or clear coat is visibly broken — scratches, scuffs, paint transfer, chips, flaking, exposed primer or bare metal, deep scrapes, gouges, keymarks |
| `other` | Something was detected on the panel but it is **not** surface paint damage — dents or creases with no paint break, panel gaps, misalignment, shadows, reflections, dirt, water marks |

**When in doubt between `scratch` and `other`, use `scratch`.** A false positive costs one extra review; a missed scratch is lost forever.

---

## Your Four Tasks

Work through each image in order:

### 1 — Add missing damage

Scan the entire image for surface damage that has no bounding box. For each area of damage you find:
- Draw a tight bounding box around it
- Assign class `scratch` or `other` per the definitions above

Focus on panel surfaces only. Ignore damage on non-panel areas (tyres, windows, number plates, scanner structure, floor, background).

### 2 — Improve inaccurate bounding boxes

Check every existing box:
- The box should hug the damage tightly — not too large, not clipping the damage
- Resize or reposition any box that is clearly misplaced or oversized
- Delete any box that is entirely on a non-panel area (open air, road surface, scanner frame, a person) with no vehicle panel visible within it at all

### 3 — Correct the class label

For each existing box, confirm the class is right:
- Visible paint break → `scratch`
- Dent, shadow, dirt, reflection, or other non-damage detection → `other`
- Change the label if it is wrong

### 4 — Flag bad-quality images

If the entire image is unusable for training, flag it as **Null** (do not annotate it at all). Remove any existing boxes first if present.

Flag as Null when:
- Image is too dark to see the panel surface
- Severe motion blur makes the panel unrecognisable
- Strong specular reflection washes out the panel entirely (no panel detail visible)
- Image is pure white, pure black, or clearly a camera fault
- The vehicle has not yet entered the frame (empty tunnel shot)

**Do not flag an image just because the damage is subtle or the lighting is imperfect.** Only flag when the image cannot be annotated with reasonable confidence.

---

## Bounding Box Quality Standards

| Rule | Detail |
|------|--------|
| Tight fit | Box edges should be within ~5% of the damage boundary — not floating in empty space |
| Single damage per box | If two separate scratches are far apart, use two boxes, not one large one covering both |
| Overlapping damage | If a scratch overlaps a dent, draw one `scratch` box around the whole affected area |
| Minimum size | Do not draw boxes smaller than roughly 10 × 10 pixels — below this size the model cannot learn from them |
| Panel only | Boxes must sit on a vehicle panel surface. Do not box damage on tyres, windows, number plates, or the scanner structure |

---

## Decision Flowchart

```
Is the image usable at all?
  └─ No (pitch-black / extreme blur / overexposed / camera fault)
       → Flag as Null, remove all boxes, move on

  └─ Yes
       ↓
Does a bounding box exist?
  └─ Yes → Does it sit on a vehicle panel?
              └─ No  → Delete it
              └─ Yes → Is the paint/clear coat broken?
                          └─ Yes → label "scratch"
                          └─ No  → label "other"

  └─ No → Is there visible damage on the panel?
              └─ Yes → Draw a box, label "scratch" or "other"
              └─ No  → Leave image with no boxes (correct — no annotation needed)
```

---

## Examples

### Keep as `scratch`
- Linear mark exposing bare metal or primer
- Paint chip where the colour coat is missing
- Scuff with paint transfer from another surface
- Deep gouge or keymark

### Relabel to `other`
- Dent or crease with paint fully intact
- Shadow or reflection that YOLO misidentified
- Dirt, mud, or water mark
- Panel gap or body misalignment

### Delete the box
- Box sits entirely on the road surface
- Box sits on open air (nothing in frame)
- Box sits on the scanner tunnel structure or a person

### Flag as Null
- Image is too dark to see anything on the panel
- Extreme motion blur — panel edges unrecognisable
- Entire frame is washed out by a bright flash

---

## Common Mistakes to Avoid

- **Do not** flag images just because the lighting is slightly dim or the damage is hard to see — make your best judgement and annotate
- **Do not** draw one large box across the whole panel to "cover everything" — each damage region gets its own box
- **Do not** use `other` for things that are clearly not on the panel at all — those boxes should be deleted entirely
- **Do not** leave the existing AI annotations unchecked — always review every box, even if it looks correct at a glance
