# PREDICTION MODELS

**Source:** Monte Carlo or Bust: Simple Simulations for Aspiring Sports Bettors

## Context

On 13 March 2020, the 2019/20 Premier League was suspended because of the COVID-19 pandemic. The last match had been played 4 days earlier. There followed what became more than a 3-month hiatus until the UK Government allowed the resumption of elite level football. The remaining 92 games were played between 17 June and 26 July. The intervening period would have offered a suitable time to build a model to predict the final outcome, including Champions League places (Liverpool were so far ahead by the time of the league suspension that they had effectively already won it) and relegation to the Championship. Model outputs could then have been compared to bookmakers' odds, with the view of exploiting any potential errors. Let's take a look at how we might have built this model.

## Modelling the outcome of the Premier League

The standard approach to predicting the outcome of a football match is to model, in probabilistic terms, the number of goals each team will score. For example, we might predict that team A has a 30% chance of scoring 1 goal, whilst team B has a 40% chance of not scoring at all. Assuming these to be independent, then using the multiplication rule we can say that the probability of team A beating team B 1-0 will be 0.4 x 0.3 = 0.12 (or 12%). If we repeat this exercise for all scores, we can then calculate the probability of a home win, a draw and an away win. For practical reasons, we will probably restrict the number of possible goals a team might score to 6 or fewer; anything more happens so rarely that it's not worth the bother of including them. Using the addition rule for mutually exclusive outcomes, the probability of a draw will be the sum of the probabilities for 0-0, 1-1, 2-2, 3-3, 4-4, 5-5 and 6-6. We can do

---

**Notes for implementation:**
- This chapter describes building a probabilistic model for football match outcomes
- Key approach: model the number of goals each team will score in probabilistic terms
- Calculate probabilities for specific scorelines (e.g., 1-0 has 12% probability)
- Aggregate scoreline probabilities to get win/draw/loss probabilities
- Practical limit: cap maximum goals at 6 per team
- Use multiplication rule for independent events
- Use addition rule for mutually exclusive outcomes (draw = sum of all tied scores)

**TODO for coding implementation:**
1. Create probability distributions for goals scored by each team
2. Calculate probabilities for all scoreline combinations (0-0 through 6-6)
3. Aggregate to home win / draw / away win probabilities
4. Compare model outputs to bookmaker odds to find +EV opportunities
