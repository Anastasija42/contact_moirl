# Cross-morphology upper-limb modeling parameters

Every value the URDF arm model needs **beyond humerus/radius length**, per morphology, **with the source next to each value** for double-checking.
Compiled 2026-06-16 from the local PDFs in this folder + literature search. See `papers/morphology_paper.tex`.

**Status tags:** `[M]` measured/published for this taxon · `[E]` estimated/derived · `[H]` no taxon-specific data — default to human (scaled).

**Human baseline** (from `human_model/urdf/human.urdf`, right-arm chain; segment lengths are subject-scaled, not the 311/230 mm osteometric means):
upper-arm segment 276 mm, forearm 287 mm, hand COM 153 mm · masses: clavicle 0.156, upper-arm 1.8, forearm 1.28, hand 0.45 kg ·
ROM: shoulder_Z ±180°, shoulder_X −60/+180°, shoulder_Y −90/+180°, elbow flexion 0–150°, elbow pron/sup −20/+180°, wrist_Z (flex/ext) ±90°, wrist_X (deviation) ±45° · humeral torsion ~140–142° · body mass ~65 kg.

---

## 1. Master table (value — source)

| Parameter | Human (ref) | Neanderthal | *H. naledi* | *A. sediba* (MH2) | *A. prometheus* (StW 573) | Chimp (*P. troglodytes*) | Bonobo (*P. paniscus*) |
|---|---|---|---|---|---|---|---|
| **Humerus** (mm) | 311 — Williams (table) | 303.9±20.2 Eur / 315.6±21.6 NE [M] — Churchill 2026 Table 8.1 p.198 | 258.0 (UW 101-283) [M] — Garvin 2017 JHE 111 p.123 | 269 [M] — Holliday 2018 Table 2 p.409 | 290.0 [M] — Heaton 2019 Table 2 p.170 | 292.7 (≤301.6) [M] — Morbeck & Zihlman 1989 Primates 30 Table 6a | ~283–285 [M] — Zihlman & Bolter 2015 |
| **Radius** (mm) | 230 — Williams (table) | 224.3±17.5 Eur / 236.3±14.1 NE [M] — Churchill 2026 Table 8.1 p.198 | not preserved → ~190 [E] (=258×0.74 human ratio) — Feuerriegel 2017 (qual.) | 226 [M] — Holliday 2018 Table 2 p.409 | 240.0 L / 250.0 R [M] — Heaton 2019 Table 3 p.173 | 271.6 [M] — Morbeck & Zihlman 1989 Table 6a | 256 [M] (Williams table) |
| **Ulna** (mm) | ~255 (human) | ~265 Eur / ~279 NE [E] (radius×1.18) — derived | — [H] | ~255–265 [E] (radius+30–40) — derived | 259.0 max / 244.0 phys. [M] — Heaton 2019 Table 4 p.175 | ~290–300 [E] — Aiello & Dean 1990 (fill) | →chimp [H] — Zihlman & Cramer 1978 note ulna differs, no value |
| **Clavicle** (mm) | 135.5 F / 150.1 M [M] — Laudicina 2023 Table 1 p.2092 | 145 F / 167.9 M / 149.5 U (n=6) [M] — Laudicina 2023 Table 1 (Trinkaus 2014) | only partial frags (73.1, 63.3, 53.6 mm); no complete length; "short", oblique [M] — Feuerriegel 2017 p.159 | **107.5** [M] — Holliday 2018 Table 2 p.409 = Laudicina 2023 Table 1 (MH2) | **142.9** (claviculohumeral ratio 49) [M] — Crompton 2021 Table 1 | 115.9 F / 123.0 M (n=15) [M] — Laudicina 2023 Table 1; 124.7 (Morbeck&Zihlman 1989) | 103 (n=1, F) [M] — Laudicina 2023 Table 1 p.2092 |
| **Hand length** (mm) | ~190 (human) | →human [H] — no data | 3rd-ray* 107.5; thumb 61.9 (=58% 3rd ray) [M] — Kivell 2015 NatComm 6:8431 (PMC4597335) | thumb (Mc1+PP1) **64**; 3rd ray (Mc3+PP3+IP3) **105** [M] — Kivell 2011 p.1415 (individual bones in SOM) | — [H] (hand not yet described) — Wikipedia/Little Foot | 237 F / 253 M (external) [M] — Schoonaert/Isler, J Anat 210 (PMC2375742) | ~234 (external) [M] — Druelle 2018 J Anat 233 Table 2 (PMC6231171) |
| **Biacromial breadth** | ~ (human) | ≈ human (scaled) [E] — PMC3970543 | no data [H] — Berger 2015 | no data; narrow inferred from short clavicle [E] — Churchill 2013 | no data; high dorsal scapula inferred [H] — Carlson 2021 | narrower/cranial, no clean mm [H] — Laudicina 2023 Anat Rec | narrower than chimp [H] — inferred |
| **Brachial index** (rad/hum×100) | 74 | 73.7 Eur / 75.5 NE [M] — Churchill 2026 Table 8.1 p.198 | n/a (no radius) | 84 [M] — Holliday 2018 | 82.8 (–86.2 R) [M] — Heaton 2019 p.191 | 93 (84–101) [M] — Morbeck & Zihlman 1989 Table 8 | ~90.5 [E] (from anchors); Pan ~92 |
| **Humeral torsion** (°) | 142.4±6.95 [M] Larson 2007; 164/161 (older convention) Larson 1988 Table 1 p.457 | retroversion 43±11.1 M / 65 F → ~137 M/~115 F on torsion conv. [E] — Churchill 2026 p.202 | **91.0** adult (UW101-283) / 105 juv [M] — Feuerriegel 2017 Table 4 p.162 | **117.0** [M] — Churchill 2013 SM **Table S8** (MH1 112.0) | **120** [M] — Heaton 2019 Table 2 p.170 | **HIGH: 148/152, range 139–159** [M] — Larson 1988 Table 1 p.457; Roach 2013 Fig 4d median ~140 | **no published value** [H] — absent from Larson 1988/2013, Roach, Selby; use high ape grade ~148 |
| **Glenoid orientation** | lateral; bar-glenoid 145° [M] — Selby&Lovejoy 2017 p.689 | anteriorly-facing, narrow/shallow [M] — Churchill 2026 p.199 | cranial; ventral bar-glenoid **121.1°** (119–122), axillospinal 26.8° [E] — Feuerriegel 2017 Table 8 p.166 | cranial; glenoid-medial (GMA) **28.8°** (human 7.1, Pan 43.9), glenoid-axillary 123.4°, glenoid-spinal 82.9° [M] — Churchill 2013 SM **Table S3** | cranial; funnel thorax; high dorsal scapula [M] — Heaton 2019 p.192–193; Carlson 2021 | cranial; bar-glenoid **126.3°**, spine angle ~100° [M] — Selby&Lovejoy 2017 p.688–689 | cranial (ape grade), no degree value [M-qual] — Arias-Martorell 2019; Young 2006 |
| **Wrist dorsiflexion cap** (°) | ~45–70 (ext) | →human [H] | →human [H] | →human; ↑ radial deviation [M-qual] — Kivell 2011 | →human [H] | **~14 typ / ~20 max** [M] — Thompson 2020 JEB 223:jeb224360 Table 3 | **~14 (5–20)** [M] — Thompson 2020 (chimp proxy) |
| **Wrist ulnar deviation** (°) | ±45 | →human [H] | →human [H] | →human [H] | →human [H] | ~25 max (arc ~21) [M] — Thompson 2020 | 20.7±2.9 ROM; −25.5 max [M] — Thompson 2020 |
| **Elbow flexion** (°) | 0–150 | →human; loading optimum at partial flexion (trochlear notch 81.8 vs 69.0) [M] — Churchill 2026 p.204 | →human [H] | →human [H] | →human; anterior trochlear notch (climbing) [M-qual] — Heaton 2019 p.175 | →human; full extension for suspension [E] — Arias-Martorell 2019 | →human [H] |
| **Forearm pron/sup** (°) | −20/+180 | →human; effective range widened (curved radius) [M-qual] — Churchill 2026 p.206–207 | →human; pronator quadratus crest developed [M-qual] — Feuerriegel 2017 | →human (Bardo 2018 pronation-efficiency, paywalled) [H] | →human; strongly curved radius/ulna [M-qual] — Heaton 2019 p.193 | →human; working arc ~13° in locomotion [M] — Thompson 2020 | →human (chimp proxy) [H] |
| **Body mass** (kg) | ~65 | 77.6 M / 66.4 F [M/E] — Wikipedia Neanderthal anatomy; energetics ScienceDirect S0047248415001049 | **37.4** (CI 34.0–40.5; small-bodied ref) or 44–46 (global ref) [M] — Garvin 2017 p.127; stature ~143.6 cm | ~26–36: MH2 ~35.5 (Holliday 2018 Table 6 p.411) vs MH2 central ~29 (range 19–44 by method), species mean 25.8 (Grabowski 2015 per-specimen table) [M] | **27–32** (femoral-head 33.2 / tibial 27.4); stature 123–125 cm [E] — Crompton 2021 p.249 (Grabowski eqns) | 47.7 F / 56.0 M (~50) [M] — Schoonaert 2007 J Anat 210 | 41.3±5.6 (34.3 F / 42.7 M) [M] — Druelle 2018; Zihlman & Bolter 2015 |
| **Upper-arm mass** (kg) | 1.8 (≈2.7% BM) | ~2.1–2.3 [E] (scale to ~77 kg + PCSA) — derived; PCSA +16–38% Churchill 2026 p.199 | ~1.03 [E] (2.7% × 38 kg) — de Leva/Winter fractions | ~1.0 [E] (~0.55× human) — body-mass scaling | ~0.8 [E] (scale to ~30 kg) — body-mass scaling | 1.91 F / 2.27 M (~3.8% BM) [M] — Schoonaert 2007 Table | 1.586 M / 1.127 F (pooled 1.389, ~3.3% BM) [M] — Druelle 2018 Table 1 |
| **Forearm mass** (kg) | 1.28 (≈1.9% BM) | ~1.3–1.5 [E] (+muscularity) — derived; Churchill 2026 p.199 | ~0.61 [E] (1.6% × 38 kg) — de Leva/Winter | ~0.7 [E] — body-mass scaling | ~0.6 [E] — body-mass scaling | 1.33 F / 1.66 M (~2.7% BM) [M] — Schoonaert 2007 | pooled 1.028 (~2.5% BM) [M] — Druelle 2018 |
| **Hand mass** (kg) | 0.45 (≈0.7% BM) | ~0.45–0.5 [E] — derived | ~0.23 [E] (0.6% × 38 kg) — de Leva/Winter | ~0.25 [E] — body-mass scaling | ~0.2 [E] — body-mass scaling | 0.64 F / 0.82 M (~1.3–1.4% BM) [M] — Schoonaert 2007 | pooled 0.595 (~1.4% BM) [M] — Druelle 2018 |
| **Strength vs human** | 1.0 | forearm flexion **+33–45%**, upper-limb PCSA +16–26% M / +19–38% F [E] — Churchill 2026 p.199, 204, 206 | robust thumb (opponens/1st DI crests); curved phalanges = climbing grip [M] — Kivell 2015 | strong flexor apparatus (climbing) + long thumb precision grip [M] — Kivell 2011 | strong brachioradialis + flexors (climbing) [M-qual] — Heaton 2019 p.172, 192 | **~1.35× dynamic muscle force/power**; ~67% fast-twitch (vs human 31–48%) [M] — O'Neill 2017 PNAS 114:7343 | upper-limb = ~36% of total muscle (vs ~20% human); forelimb ~14% BM [M] — Zihlman & Bolter 2015 PNAS 112:7466 |

\* **naledi "hand length" caveat:** 107.5 mm is the **third-ray** length (Mc3+PP3+IP3) from Kivell 2015, *not* a wrist-to-fingertip total — not directly comparable to the chimp/bonobo "external hand length" (234–253 mm, a skin-surface measure). Treat as different metrics.

† **Neanderthal torsion caveat:** sources report **retroversion** (43°±11.1 M, 65° F; Churchill 2026 p.202). Torsion and retroversion are complementary conventions; the ~137°/~115° figures are the mapping onto the human ~140° torsion convention, *not* directly published torsion values.

---

## 2. Per-species detail & double-check notes

### Neanderthal (*H. neanderthalensis*) — local PDF: `Churchill, 2026.pdf`
- Humerus/radius/BI: Table 8.1, **p.198** (Europe n=7, Near East n=7).
- Retroversion 43°±11.1 (R males, n=6); 65° (females, n=2); bilateral asymmetry low ~2.8° — **p.202**.
- Glenoid: large, supero-inferiorly tall, AP-narrow/shallow, anteriorly directed — **p.199**; 3D analysis ScienceDirect `S1040618216316299`.
- Elbow: trochlear notch index 81.8±7.3 (vs human 69.0±5.2); long olecranon (triceps MA 0.091 vs 0.061–0.066) — **p.204–205**.
- Forearm strength: biceps MA 0.153 (vs 0.134–0.147), brachialis 0.164 (vs 0.142–0.155) → +33–45% flexion; PCSA +16–26% M / +19–38% F — **p.199, 204, 206**.
- Deltoid tuberosity small (weak elevated-arm/throwing) — **p.200**.
- No published per-segment masses or ROM-in-degrees → default human, scaled to ~77 kg.

### *Homo naledi* — no local PDF (paywalled body text)
- Humerus 258.0 mm, body mass **37.4 kg** (CI 34–40.5), stature ~143.6 cm — Garvin 2017 JHE 111 **p.123–126** (open PDF: wordpressua.uark.edu/.../Garvin-et-al_in-press.pdf).
- Humeral torsion **91°** adult (UW 101-283), 105° juvenile — Feuerriegel 2017 JHE 104:155–173 (PMID 27839696); cross-check johnhawks.net fossil-profile post. **Full text paywalled — exact glenoid angle + clavicle index in tables ~pp.158–165 not openly retrievable.**
- Hand: 3rd-ray 107.5 mm, thumb 61.9 mm (58% of 3rd ray); doubly-curved phalanges (climbing); carpus ~human-like — Kivell 2015 NatComm 6:8431 (open: PMC4597335).
- Glenoid cranial; scapula high/lateral; clavicle short — Feuerriegel 2017; Berger 2015 eLife 4:e09560.
- No segment masses → scale ~38 kg with de Leva/Winter (upper-arm 2.7%≈1.03, forearm 1.6%≈0.61, hand 0.6%≈0.23 kg).

### *A. sediba* (MH2) — local PDF: `Holliday et al., 2018.pdf`
- Humerus 269, radius 226, clavicle 107.5 mm; MH1 humerus 248 mm — Holliday 2018 Table 2 **p.409**.
- Body mass 30–36 kg (MH1 ~35, MH2 ~35.5) — Table 6 **p.411**.
- Humeral torsion **117°** — Churchill et al. 2013 Science 340:1233477 (paywalled; via ResearchGate 236196987).
- Hand: longest thumb relative to fingers of any hominin; precision-grip + strong flexors — Kivell 2011 Science 333:1411 (`10.1126/science.1202625`, in `references_morphology.bib`).
- Cranial glenoid, cranial scapular spine, enlarged supraspinatus — Churchill 2013.
- Forearm pronation efficiency: Bardo et al. 2018 AJPA (`10.1002/ajpa.23319`) — **paywalled, ROM degrees not extracted**.
- No segment masses → scale ~35 kg (~0.55× human).

### *A. prometheus* / StW 573 "Little Foot" — local PDF: `Heaton et al., 2019.pdf`
- Humerus 290.0 (Table 2 p.170), radius 240 L/250 R (Table 3 p.173), ulna 259 max/244 phys (Table 4 p.175).
- Humeral torsion **120°** (Larson method), Table 2 **p.170**, context Fig. 2 p.171.
- Glenoid cranial, funnel (apically-narrow) thorax, long curved clavicle, high dorsal scapula — Heaton **p.192–193**; Carlson et al. 2021 JHE (PMID 33888323).
- Strong brachioradialis (flaring lateral supracondylar ridge), strong flexors; ulnar keeling intermediate (climbing) — **p.172, 177, 192**.
- Radius/ulna strongly curved (climbing or antemortem trauma debate) — **p.169, 193**.
- Body mass not published → ~30 kg (McHenry 1992 small-female australopith mean, PMID 1580350); stature ~1.20–1.30 m (Clarke). No segment masses, no ROM degrees.

### Chimpanzee (*P. troglodytes*)
- Humerus 292.7, radius 271.6, clavicle 124.7 mm, BI 93 — Morbeck & Zihlman 1989 Primates 30, Tables 6a/8/9 (PDF: apeanatomyevolution.com/.../1989-Primates-Gombe-chimp-size.pdf).
- **Humeral torsion HIGH ~130–145° (≈ human)** — Larson et al. 2007 JHE (PDF: whereareyouquetzalcoatl.com/.../HFloresiensisLarsonEtAl2007.pdf); exact Pan mean paywalled (Larson 1996/2013). **NOT low — correct any code that says otherwise.**
- Glenoid cranial, shallow ~6 mm — Selby & Lovejoy 2017 AJPA 162 (`10.1002/ajpa.23158`); MacLean & Dickerson 2020 JEB 223; Vermeulen 2023 J Anat 242 (PMC9877474, angle deltas).
- **Wrist extension ~14° typ / ~20° max; ulnar deviation ~25°** — Thompson 2020 JEB 223:jeb224360 Table 3 (the knuckle-walking dorsiflexion ceiling).
- Segment masses: upper-arm 1.91 F/2.27 M, forearm 1.33/1.66, hand 0.64/0.82 kg; body 47.7 F/56.0 M kg; COM upper-arm 52%/forearm 55%/hand 52% proximal — Schoonaert, D'Août & Aerts 2007 J Anat 210:518–531 (PMC2375742); inertia cross-check Isler 2006 J Anat 209 (PMC2100316).
- Strength **1.35× dynamic** muscle force/power; ~67% fast-twitch; fibres 59% of MTU length (vs human 44%) — O'Neill 2017 PNAS 114:7343 (PMID 28652350). (Abstract's "~1.5×" is the whole-organism literature estimate — don't conflate with the 1.35× model result.)

### Bonobo (*P. paniscus*)
- Humerus ~283–285, radius 256 mm; IMI 100±5.3 (vs chimp 105) — Williams table; Zihlman & Bolter 2015.
- **No bonobo-specific torsion / ROM** → chimp proxy (Druelle 2018 shows bonobo–chimp differences subtle/size-driven). Torsion → ~140 (high African-ape grade); wrist/ROM → Thompson 2020 chimp values.
- Glenoid cranial (ape grade) — Arias-Martorell 2019 (PMC6342098); scapula smaller/longer/narrower than chimp — Young 2006 (PMC2100339).
- Segment masses **bonobo-specific**: upper-arm pooled 1.389 (1.586 M/1.127 F), forearm 1.028, hand 0.595 kg; body 41.3±5.6 kg; whole forelimb ~14% BM — Druelle et al. 2018 J Anat 233(6):843–853 Tables 1–2 (PMC6231171).
- Muscle: upper-limb ~36% of total muscle (vs ~20% human) — Zihlman & Bolter 2015 PNAS 112:7466 (PMC4475937); thumb PCSA only 10.6% of forearm (vs 17.5% human) — van Leeuwen 2018 J Anat 233:328 (PMC6081514).
- Tool use: Kanzi/Pan-Banisha percussive flaking — Roffman et al. 2012 PNAS 109:14500 (`roffman2012stone` in bib).
- **Reference corrections:** forelimb-muscle companion to Payne 2006 (hindlimb) is **Myatt et al. 2012** J Anat 220:13; Oishi studied orang+chimp, **not** bonobo; correct Druelle cite is **J Anat 233(6):843–853**.

---

## 3. Modeling guidance (consequences of the above)
1. **Fossils (naledi, sediba, StW 573):** no segment masses published anywhere → scale **human mass fractions** (de Leva/Winter) to body mass; do **not** use absolute human kg (over-masses a 30–40 kg body).
2. **Apes (chimp, bonobo):** segment masses **are** published — use directly (Schoonaert 2007 / Druelle 2018), do not scale from human.
3. **Shoulder rpy:** every non-human here (apes + all australopiths + naledi) has a **cranially oriented glenoid** → tilt glenoid cranially, in addition to torsion.
4. **Humeral torsion** is the *early-hominin* signal (naledi 91°, sediba 117°, StW 573 120°), **not** an ape signal (chimp ~140° ≈ human). Bonobo has no value → high African-ape grade.
5. **Pan wrist dorsiflexion cap ~14–20°** (Thompson 2020) is the one firm ape-specific ROM departure; all other ROM degrees default to human across taxa (no fossil ROM-in-degrees exists).

## 4. Values recovered from the obtained paywalled PDFs (`~/Desktop/paywall/`, read 2026-06-16)

**RESOLVED:**
- ***Pan* humeral torsion is HIGH, confirmed numerically:** *P. troglodytes* **148°/152°** (range 139–159), *Gorilla* 163/165°, *Homo* 164/161°, *Pongo* 141/153°, *Hylobates* ~128–137° — Larson 1988 Table 1 p.457 (Evans&Krahl/Zapfe data). Roach 2013 Fig 4d: Pan median ~140°. **No bonobo (*P. paniscus*) torsion value exists in any source** — the "*A. paniscus*" in Larson's table is the spider monkey *Ateles*, not the bonobo.
- **naledi torsion 91.0°** (UW 101-283 adult), 105° juvenile — Feuerriegel 2017 Table 4 p.162. **Glenoid:** ventral bar-glenoid angle 121.1° (range 119–122), axillospinal 26.8° → cranial — Table 8 p.166. **Humerus 256.0 mm**, robusticity index 0.18. **No complete forearm/clavicle/body-mass** in this paper (fragments only; largest radius frag 192.5 mm).
- **Clavicle lengths (Laudicina 2023 Table 1 p.2092):** human 135.5 F/150.1 M; chimp 115.9 F/123.0 M; **bonobo 103 (n=1 F)**; gorilla 134.5 F/166.8 M; orang 155.1 F/171.0 M; *A. afarensis* 156.4 M (KSD); **A. sediba 107.5 F (MH2)**; *H. erectus* 132 M; **Neandertal 145 F/167.9 M (n=6)**. (Same table also gives body masses, useful for mass scaling.)
- **sediba hand (Kivell 2011 main text p.1415):** thumb (Mc1+PP1) 64 mm; 3rd ray (Mc3+PP3+IP3) 105 mm; brachial index 84 (Churchill 2013 p.4).
- **sediba/afarensis glenoid (Selby&Lovejoy 2017 p.688–689):** MH2 glenoid-medial angle 28.8°, GLENVERTANG 5.4° (human-like, NOT suspensory-cranial despite ape-like torsion); bar-glenoid chimp 126.3°, gorilla 131.6°, orang 127.4°, human 145°, A.L.288-1 130°.
- **Forearm pronation (Ibáñez-Gimeno 2017):** gives NO pron/sup ROM in degrees — uses pronation *efficiency* (cm). Useful geometry (Table 2 p.793): MH2 radius physiological length 224 mm, radial head radius 0.91 cm, carrying angle λ = **−6.0°** (A.L.288-1 +2.5°, human +8.79°, hylobates +13.4°); radial curvature AO′ MH2 2.54 cm.

## 5. Supplementary-material values (obtained 2026-06-16: Churchill 2013 SM, Kivell 2011 SOM, Myatt 2011 App. A)

### 5a. *A. sediba* MH2 — Churchill et al. 2013 Supplementary Material (all n=1, [measured])
- **Humeral torsion — Table S8:** **MH2 = 117.0°**, MH1 = 112.0°. Convention: 90° = posterior-facing head, 180° = medial; human 165.0°±6.9 (n=27). In this *single consistent frame*: Hylobates 116.6, **A. sediba 117**, A. afarensis (A.L.288-1) 124, H. erectus 110–111.5, H. floresiensis 115, Pongo 135, **Pan troglodytes 153.4**, Gorilla 159.9, Homo 165. → sediba is gibbon-grade low, ~48° below human. (See convention note in §3.)
- **Scapular angles — Table S3:** glenoid-medial GMA **28.8°** (human 7.1, Pan 43.9, Gorilla 31.5, Pongo 26.2 — MH2 ape-like, near Pongo/Gorilla); glenoid-axillary GAA 123.4°; glenoid-spinal GSA 82.9°; spinal-medial SMA 56.2°; spinal-axillary SAA 28.2°; axillary-vertebral AVA 42.1°. (Note: the older "114° axilloglenoid" of Kibii 2011 ≈ GAA under a different landmark.)
- **Long bones — Tables S1/S9/S11/S12:** clavicle (max length) **107.5 mm**; humerus (max) **269 mm**; radius (articular length RAL) **220 mm**; ulna (articular length UAL) **223 mm**; brachial index **84**; claviculohumeral index 40.0; trochlear-notch orientation 84.0 (ape/climbing-like, vs human 70.6); brachialis MA 0.127, triceps MA 0.075.

### 5b. *A. sediba* MH2 hand bones (mm) — Kivell et al. 2011 SOM Table S2 ([measured], right hand)
Mc1 **39.4**, Mc2 **52.8**, Mc3 **48.3**, Mc4 **44.0**, Mc5 **41.6** · PP1 **24.1** (L; right est. [23.8]), PP2 **31.2**, PP3 **34.7**, PP4 **33.4**, PP5 **27.2** · IP2 [16.4] est., IP3 **21.6**, IP4 **20.4**, IP5 **16.8** · DP1 (thumb) **15.1**.
Thumb (Mc1+PP1)=63.5≈64; 3rd ray (Mc3+PP3+IP3)=104.6≈105; thumb/ray-3 ratio **60.8%** (human 54.0%) — Table S14. Carpus: capitate **17.7**, hamate 16.6–18.5, scaphoid PD 11.7 mm. Wrist angle: capitate Mc2–Mc3 facet **117°** (human 134°, apes 85–86°) — Table S11; Mc2-trapezium facet 21°. Phalangeal curvature: only graphical (Fig S11), "intermediate, like Au. robustus", ~0.03 Ax2; no tabulated coefficient.

### 5c. Chimpanzee forelimb muscle — Myatt 2011 Appendix A (subject **Ptsm**, *P. troglodytes* ♂ 50.2 kg) — mass(g)/PCSA(cm²) [measured, n=1]
Deltoid 369.5/48.4 · Latissimus dorsi 583/21.9 · Subscapularis 206/32.4 · Supraspinatus 75/16.1 · Infraspinatus 26/3.8 · Teres major 191/13.1 · **Triceps brachii 400/51.0** · **Biceps brachii 275/18.5** · Brachialis 191/14.1 · Brachioradialis 113/7.1 · Supinator 52/14.4 · Pronator teres 52/15.3 · Pronator quadratus 13/5.8 · Flexor carpi ulnaris 77/17.7 · Flexor carpi radialis 81/12.7 · **Flexor digitorum profundus 208.5/28.1** · **Flexor digitorum superficialis 166/41.2** · Ext. carpi ulnaris 33/6.4 · Ext. carpi radialis longus 46/2.7 · Ext. carpi radialis brevis 42/6.3 · Ext. digitorum communis 57/8.0 · Abductor pollicis longus 28/5.7 · Trapezius 239/22.6. (Largest PCSAs: triceps, deltoid, FDS — power extension + grip. Pectoralis major missing in all specimens.)
**Bonobo (Ppam) is NOT in Appendix A** — it lists only Ptsm + 6 gorillas + 3 orangs. For bonobo per-muscle data use Druelle 2018 (segment masses) + the pooled great-ape allometric constants (Myatt Tables 3–5); bonobo n=1 anyway.

### 5d. Bonobo muscle composition — Diogo 2018 (Front. Ecol. Evol. 6:53) [qualitative — no PCSA/mass]
This review gives **no quantitative muscle architecture** (no mass/PCSA/fascicle). It does establish: (i) bonobo and common-chimp forelimb musculature are **essentially identical in composition** — the only consistent forelimb difference is intermetacarpales 1–4 fusing with flexores breves profundi to form dorsal interossei 1–4 (human-like) → **chimp PCSA/mass is a justified bonobo proxy**; (ii) bonobos have a **stout, non-vestigial flexor digitorum profundus tendon to the thumb** in all 7 specimens (unlike chimp/gorilla/orang), i.e. a functional long thumb-flexor — supports bonobo manipulation capability. **Quantitative bonobo muscle WEIGHTS are in Diogo et al. 2017b** *Photographic and Descriptive Musculoskeletal Atlas of Bonobos … and Weight of the Muscles* (Springer) — the book, not this review; also Diogo et al. 2017a (Sci. Rep. 7:608).

### 5e. Bonobo muscle masses — Diogo et al. 2017b atlas (`bonobo_atlas.pdf`) [measured; WEIGHT only, no PCSA/fascicle]
Per-muscle weights given for adult **Kidogo**, infant Foyo, fetus Ano. **Adult upper-limb weights cover only the PROXIMAL (shoulder/arm) muscles** — the atlas does NOT weigh the adult's forearm/hand muscles (pronators, supinator, brachioradialis, wrist/digital flexors-extensors, thenar give infant+fetus weights only). So bonobo distal-muscle masses still need a proxy (Myatt chimp PCSA §5c, or scale the infant).
**Adult Kidogo proximal masses (g):** deltoideus 222.76, biceps brachii 173.50, triceps brachii 353.87, brachialis 132.95, coracobrachialis 46.22, subscapularis 157.39, supraspinatus 46.96, infraspinatus 92.02, teres major 161.06, teres minor 23.16, latissimus dorsi 493.12, dorsoepitrochlearis 43.20, pectoralis major 301.68, pectoralis minor 22.30, serratus anterior 170.28, rhomboideus 78.51, levator claviculae 10.78, subclavius 1.26. (≈60–85% of the Myatt chimp masses in §5c → chimp proxy validated; bonobo smaller-bodied.)
**Note:** atlas gives mass only → for force capacity (PCSA) combine these masses with fascicle-length data (Myatt 2011 pooled great-ape allometry, §5c source).

### 5f. Bonobo single-fiber contractile properties — Degens et al. 2025 (*J. Exp. Zool. A*) [measured; the force-capacity multiplier]
Single-fiber force-velocity study, bonobo vs human, by fiber type (Table 2). **This supplies the missing intrinsic force term: specific tension.** whole-muscle force = PCSA × specific tension.
| Property | Bonobo I / II | Human I / II |
|---|---|---|
| Specific tension (N·cm⁻²) | 10.7 / 12.1 | 14.1 / 13.7 |
| Vmax (FL·s⁻¹) | 0.295 / 0.770 | 0.373 / 0.939 |
| Fiber CSA (µm²) | 7648 / 9654 | 5840 / 7037 |
| Po single-fiber (µN) | 750 / 1116 | 791 / 966 (≈ no diff) |
| a/Po (F-V curvature) | 0.119 / 0.195 | 0.066 / 0.099 |
| Specific power (W·L⁻¹) | 1.99 / 7.73 | 2.03 / 8.09 (≈ same) |

**KEY for the effort model — apes are NOT intrinsically stronger per PCSA.** Bonobo specific tension is ~76% (type I) / 88% (type II) of human, Vmax also lower; single-fiber Po is similar only because bonobo fiber CSA is larger. *"The 'super strength' of bonobos cannot be explained by a higher specific tension"* — the advantage is architectural (PCSA), and power parity comes from higher a/Po curvature. Degens notes O'Neill 2017's chimp data also suggest **lower** specific tension. → For chimp AND bonobo: keep specific tension at human-or-slightly-lower (~11–14 N·cm⁻²) and let **PCSA/architecture** carry any force difference; do not scale specific tension up. This refines the prior O'Neill "~1.4× dynamic" note (that 1.35× is whole-muscle dynamic power from architecture + fast-fiber %, NOT per-PCSA fiber strength). Source: "A Comparison of the Force-Velocity Relationship of Bonobo and Human Muscle Fibers", Degens et al., `~/Desktop/paywall/`.

### 5g. StW 573 (*A. prometheus*) — Crompton et al. 2021 (*Folia Primatol.* 92:243–275, DOI 10.1159/000519723) [measured/estimated]
Comprehensive functional synthesis; fills most StW 573 gaps beyond Heaton 2019.
- **Clavicle = 142.9 mm** (Table 1 p.251), AP diam at conoid 14.3, midshaft circ 38, SI diam 11.6 mm; **claviculohumeral ratio 49**.
- **Body mass 27–32 kg** (femoral-head SI eqn 33.2 kg; tibial-distal-ML eqn 27.4 kg; Grabowski 2015 eqns) — p.249. **Stature 123–125 cm** (from femur max length 335 mm, Hens 2000 pygmy eqns).
- Limb lengths (Heaton 2019): humerus 290, radius 240–250, **ulna 259** mm. Femur 335 mm.
- **Brachial index 82.8, intermembral 85.5, humerofemoral 86.6** (p.249–254); ape-intermediate.
- **Humeral torsion 120°** (p.252) — lower than AL 288-1/Sts 7, slightly higher than MH2 117°.
- **Glenoid cranially oriented** (more than human), large ape-like supraspinous fossa, dorsally-positioned scapula, stout ventral bar — qualitative, **no degree value** (refers Carlson 2021).
- Upper-limb muscle/robusticity: strong lateral supracondylar crest (brachioradialis → climbing power in pronation+elbow flexion); medially-oriented radial tuberosity (biceps supination power); brachialis MA 0.139, triceps MA 0.068 (less elbow-extension power than other australopiths); strongly curved radius+ulna (climbing).
- **Hand:** "virtually complete hand" preserved, modern-human-like thumb/finger proportions + precision-grip distal thumb phalanx — but **individual hand-bone lengths deferred to Jashashvili et al. (under review)**, NOT in this paper.

## 6. Remaining gaps (genuinely not yet in hand)
- **naledi:** no complete forearm bone (fragments only; radius est. ~190 mm), no complete clavicle, no body mass in Feuerriegel 2017 → body mass 37.4 kg from Garvin 2017.
- **StW 573:** body mass/stature/clavicle now resolved (Crompton 2021, §5g); only **hand-bone lengths** remain unpublished (Jashashvili et al. under review).
- **Chimp ulna length** (use Aiello & Dean 1990 / Schultz); **biacromial breadth** for apes/fossils (no per-taxon means exist; Laudicina 2023 gives human NHANES + graphical chimp ~240 mm ♂).
- **All taxa:** joint ROM in degrees is unpublished except the *Pan* wrist (Thompson 2020); default to human, apply the cranial-glenoid + torsion offsets + the *Pan* ~14–20° dorsiflexion cap.
