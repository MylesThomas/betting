# Paint Scoring Visualization - Gradient Explanation

## Color Gradients Used

### 1. **Paint FG%** (Performance Quality)
**Color Scale:** 🔴 Red → ⚪ White → 🟢 Green

| Color | FG% Range | Meaning |
|-------|-----------|---------|
| 🟢 Dark Green | 70-75% | Elite finisher (Luka, Ant Edwards, Pritchard) |
| 🟢 Light Green | 65-70% | Very good finisher |
| ⚪ White | 60-65% | Average finisher |
| 🔴 Light Red | 55-60% | Below average finisher |
| 🔴 Dark Red | 50-55% | Poor finisher (Darius Garland, Jordan Poole) |

**Domain:** 50% to 75%  
**Purpose:** Shows **efficiency** - how well they finish when they get to the paint

---

### 2. **Paint Rate** (Volume)
**Color Scale:** ⚪ Light Gray → 🔵 Dark Blue

| Color | Rate Range | Meaning |
|-------|------------|---------|
| 🔵 Dark Blue | 35%+ | Frequent paint attacker (Ant Edwards 32.7%, Tyrese Maxey 37.5%) |
| 🔵 Medium Blue | 25-35% | Regular paint attacker |
| 🔵 Light Blue | 15-25% | Occasional paint scorer |
| ⚪ Gray | 0-15% | Rarely attacks paint (Pritchard 14.3%) |

**Domain:** 0% to 50%  
**Purpose:** Shows **how often** they take paint shots as % of all their shots

**Key Insight:** Pritchard's 14.3% rate is LOW (light blue) - he's selective about when he attacks the paint, but when he does, he's ELITE (67.1% FG%)

---

### 3. **Paint PPG** (Scoring Output)
**Color Scale:** ⚪ Light Orange → 🟠 Dark Orange

| Color | PPG Range | Meaning |
|-------|-----------|---------|
| 🟠 Dark Orange | 8-10 | High volume paint scorer (Ant 8.9, Maxey 9.9) |
| 🟠 Medium Orange | 5-8 | Moderate paint scorer |
| 🟠 Light Orange | 2-5 | Low volume paint scorer (Pritchard 3.0) |
| ⚪ Gray | 0-2 | Minimal paint scorer |

**Domain:** 0 to 10 points per game  
**Purpose:** Shows **total scoring impact** in the paint

---

## Why These Gradients Matter

### Pritchard's Profile:
- **Paint FG%:** 67.1% (🟢 Light Green) = **ELITE efficiency**
- **Paint Rate:** 14.3% (⚪ Light Blue) = **Low volume** (selective)
- **Paint PPG:** 3.0 (🟠 Light Orange) = **Low output** (doesn't attack much)

### Interpretation:
**"When Pritchard gets to the paint, he's elite... but he doesn't get there often."**

This makes him UNIQUE compared to:
- **Tyrese Maxey:** High volume (37.5% rate), but worse efficiency (59.1%)
- **Anthony Edwards:** High volume (32.7% rate), elite efficiency (68.4%), high output (8.9 PPG)

---

## Comparison: 3 Different Player Types

### Type 1: **Elite Efficiency + High Volume** (Anthony Edwards)
- Paint FG%: 68.4% (🟢 green)
- Paint Rate: 32.7% (🔵 dark blue)
- Paint PPG: 8.9 (🟠 dark orange)
- **Profile:** Dominant paint scorer - attacks often AND finishes well

### Type 2: **Elite Efficiency + Low Volume** (Payton Pritchard)
- Paint FG%: 67.1% (🟢 green)
- Paint Rate: 14.3% (⚪ light blue)
- Paint PPG: 3.0 (🟠 light orange)
- **Profile:** Selective but deadly - picks his spots perfectly

### Type 3: **Poor Efficiency + High Volume** (Darius Garland)
- Paint FG%: 50.0% (🔴 red)
- Paint Rate: 26.4% (🔵 medium blue)
- Paint PPG: 3.8 (🟠 light orange)
- **Profile:** Struggles to finish despite attacking frequently

---

## The Story for Your Post

**"Pritchard is the most EFFICIENT small guard in the paint (67.1%), but he's selective about when he attacks (14.3% rate)."**

This is what makes him special:
1. He's not an explosive athlete like Ja Morant
2. He doesn't attack as often as Maxey or Brunson
3. But when he DOES attack, he's #1 among guards 6'2" and under

**It's all craft, timing, and IQ - not athleticism.**

---

## Technical Notes

- **Palette:** 5-color gradients for smooth transitions
- **Domain:** Sets the min/max values for the color scale
- **NA Color:** Gray (#e8e8e8) for missing data
- **Method:** "numeric" for continuous color mapping

Colors chosen to match:
- Green/Red (universal for good/bad)
- Blue (neutral, shows volume)
- Orange (warm, shows scoring output)

