# Monte Carlo or Bust: Simple Simulations for Aspiring Sports Bettors

**Book by:** Joseph Buchdahl

---

# PREDICTION MODELS

**Source:** Monte Carlo or Bust: Simple Simulations for Aspiring Sports Bettors

**Chapter:** Prediction Models

================================================================================


--- Page 1 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

PREDICTION MODELS

On 13 March 2020, the 2019/20 Premier League was suspended because of the COVID-19
pandemic. The last match had been played 4 days earlier. There followed what became more
than a 3-month hiatus until the UK Government allowed the resumption of elite level football.
The remaining 92 games were played between 17 June and 26 July. The intervening period
would have offered a suitable time to build a model to predict the final outcome, including
Champions League places (Liverpool were so far ahead by the time of the league suspension
that they had effectively already won it) and relegation to the Championship. Model outputs
could then have been compared to bookmakers’ odds, with the view of exploiting any potential
errors. Let’s take a look at how we might have built this model.

Modelling the outcome of the Premier League

The standard approach to predicting the outcome of a football match is to model, in
probabilistic terms, the number of goals each team will score. For example, we might predict
that team A has a 30% chance of scoring 1 goal, whilst team B has a 40% chance of not scoring
at all. Assuming these to be independent, then using the multiplication rule we can say that
the probability of team A beating team B 1-0 will be 0.4 x 0.3 = 0.12 (or 12%). If we repeat this
exercise for all scores, we can then calculate the probability of a home win, a draw and an away
win. For practical reasons, we will probably restrict the number of possible goals a team might
score to 6 or fewer; anything more happens so rarely that it’s not worth the bother of including
them. Using the addition rule for mutually exclusive outcomes, the probability of a draw will
be the sum of the probabilities for 0-0, 1-1, 2-2, 3-3, 4-4, 5-5 and 6-6. We can do similar for the
home and away win probabilities. Finally, you may recall that to convert to fair decimal betting
odds (before a margin has been applied), we simply invert the decimal probabilities.

But how do we calculate the probability of a team scoring a certain number of goals?
One well documented approach, first published by Mark Dixon and Stuart Coles (of Lancaster
University) in the journal of Applied Statistics in 1997 (Volume 46, Issue 2), develops the
concept of attack and defence strength, by comparing individual teams goal scoring and
conceding to the league averages (means) over a specified number of previous games. These are
then used to estimate the expected number of goals each team will score in a game. Finally,

1 minute left in chapter

the Poisson distribution is used to calculate the probabilities of individual goal tallies, where
the expected number of goals is the distribution’s mean. Recall that ‘expected’ and ‘mean’ are
effectively the same thing. One area of debate is how many games one should include, over
which the averages should be determined. Too many, and the data may not be relevant for a
team's current strength, while too few may allow outliers to bias the data. For my model, I will
use the games played in the 2019/20 Premier League season up to the point of suspension.
Furthermore, I will use the xG metric instead of actual goals to further reduce the influence of
luck in goal scoring over that time, which will hopefully make it more reliable. Let’s pick the
first game after the resumption of the League - Aston Villa v Sheffield United on 17 June 2020-
as an example to illustrate the process.

To calculate attack strength, the first step is to calculate the Premier League mean xG per
game. Since home teams, on average, perform better than away teams, we should consider
them separately for this purpose. Indeed, pretty much all football prediction modelling does
so. Prior to the League suspension, there were 288 games played and 455.94 xG, as calculated
by Understat.com, attributed to home teams. Hence the mean Premier League home team xG
scored during this period was 455.94/288 = 1.583 xG. Similarly, for away teams, with a total
of 384.97 xG, the mean scored was 1.337 xG. The next step is to calculate the mean xG for
both Aston Villa and Sheffield United. In their 13 home games prior to suspension, Aston Villa
amassed 17.96 xG. Consequently, their mean home xG was 1.382. Sheffield United amassed
13.33 xG in their 13 away games, implying their mean away xG was 1.025. Finally, to calculate
team’s attack strength, divide the home/away team’s mean xG scored by the Premier League’s
mean home/away xG scored. For Aston Villa this is 1.382/1.583 = 0.873. For Sheffield United it
is 1.025/1.337 = 0.767. The larger the attack strength number, the stronger the attack.

The defence strength is calculated in exactly the same way, using xG conceded instead of xG
scored. Unsurprisingly, the Premier League mean xG conceded is just the reverse of the mean
xG scored for home and away teams respectively, since goals scored by the away team are goals
conceded by the home team, and vice versa. Thus, home teams conceded a mean of 1.337 xG
prior to 13 March 2020, with the away teams conceding a mean of 1.583 xG. With Aston Villa
conceding 27.04 xG in 13 home games (mean = 2.08 xG) and Sheffield United conceding 18.83
xG in 13 away games (mean = 1.448 xG), the defence strengths for the two teams are 1.556
and 0.915 respectively. This time, the smaller the defence strength number, the stronger the
defence.

Finally, we are in a position to calculate the expected number of goals Aston Villa and
Sheffield United will score in their match. For Aston Villa, this is done by multiplying Aston
Villa’s home attack strength (0.873) by Sheffield United’s away defence strength (0.915) and the
mean number of home xG scored in the Premier League (1.583). The answer is 1.264 xG. For
Sheffield United, we multiply their away attack strength (0.767) by Aston Villa’s home defence
strength (1.556) and the mean number of away xG scored in the Premier League (1.337). The
answer is 1.596 xG. Thus, if our model is correct, and if this game could be played an infinite

14%




--- Page 2 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

number of times to eliminate the influence of the random variables, we would expect Aston
Villa to score an average of 1.264 goals and Sheffield United to score an average of 1.596 goals.
On that basis, it looks like the away team would be favourites to win. The bookmakers certainly
thought so, pricing Sheffield United at 2.32 on average (the top bookmaker offered 2.43), whilst
making Aston Villa 3.28 (with a best price of 3.50). For the record, the draw was priced at 3.29
(best 3.44).

The Dixon-Coles model gives us the expected goals in a football match. It further assumes
that the scoring of goals is independent, that is to say one goal does not cause another to be
scored, and that they are Poisson distributed. We can then use these assumptions to calculate
the probability of any goal tally for either team. I introduced the Poisson distribution to you
earlier in my primer on statistics and showed how the distribution of home and away goals
conforms quite closely to the Poisson distribution. How did I calculate those Poisson goal tallies
in my comparison chart? Knowing that the mean of a Poisson distribution is equal to the
variance, it is quite straightforward to calculate the probabilities for any goal tally. Well, it is
for a trained statistician. Fortunately, we don’t need to worry about learning the mathematics.
Excel will do the job for us with its POISSON(x,u,,cumulative) function. If we know the mean,
we can use the function to calculate the expected probability of any goal tally. Here, x is the
number of goals for which we want to calculate the probability, u is the mean or expected
number of goals calculated by the Dixon-Coles model (recall that ‘expected’ and ‘mean’ are
essentially the same thing), and ‘cumulative’ denotes a logical argument that determines the
form of the probability distribution returned. If cumulative is TRUE, the POISSON function
returns the cumulative Poisson probability that the number of goals events occurring will be
between zero and x inclusive; if FALSE, it returns the Poisson probability that exactly x goals
will be scored. For example, the probability of Aston Villa scoring exactly 3 goals calculated via
this function is POISSON(3,1.264,FALSE) = 9.51%. Using Excel’s POISSON function and the two
teams’ expected goals, as estimated by Dixon-Coles, I’ve calculated the Poisson probabilities for
Aston Villa’s and Sheffield United’s possible goal tallies. The results are shown in the table below
(Poisson %) up to a maximum of 6 goals. Knowing the probabilities for either team to score a
specified number of goals, we can then use the multiplication rule to calculate the probabilities
for any score. The correct-score probabilities are also shown in the shaded area.

1 minute left in chapter

SHU 0 1 2 3 4 5 6
AV Poisson % 20.27% 32.35% 25.82% 13.73% 5.48% 1.75% 0.47%
0 28.25% | 3.73% 9.14% 7.29% 3.88% 1.55% 0.49% 0.13%
1 35.71% | 7.24% 11.55% 9.22% 4.90% 1.96% 0.62% 0.17%
2 = 22.57% 457% 7.30% 5.83% 3.10% 1.24% 0.39% 0.11%
3 9.51% 1.93% 3.08% 2.45% 1.31% 0.52% 0.17% 0.04%
4 3.00% 0.61% 0.97% 0.78% 0.41% 0.16% 0.05% 0.01%
5 0.76% 0.15% 0.25% 0.20% 0.10% 0.04% 0.01% 0.00%
6 0.16% 0.03% 0.05% 0.04% 0.02% 0.01% 0.00% 0.00%

Finally, we can use the additional rule to calculate the probabilities of the three possible results:
home win, draw or away win. The Aston Villa home win probability is calculated by summing
the darker shaded area; the Sheffield United away win probability is calculated by summing the
lighter shaded area; and the draw probability is the sum of the probabilities along the diagonal.
These come to 30.24%, 45.00% and 24.59% respectively. You will note that these sum to
99.83%. The missing 0.17% will be found in scores involving 7 or more goals for either team.
To calculate the fair odds as estimated by this model, we invert the probabilities. If you want
you can do this for the individual correct scores, but here I will just do that for the match result.
Aston Villa to win would be priced at 3.31, Sheffield United at 2.22 and the draw 4.07.

Comparing these to the best available prices from the bookmakers on the day of the match
I reported above, we would conclude that there was expected value backing both Aston Villa
(3.50/3.31 — 1 = 5.74%) and Sheffield United (2.43/2.22 — 1 = 9.46%). In contrast, the draw
was a negative expectation bet (3.44/4.07 - 1 = -15.48%). In the event, the game finished 0-0
(a 5.73% chance according to the model). Does that mean the model was wrong? Of course not.
Yes, the model could be wrong, but a single result won't tell you that. Remember the random
variables that influence the match. Luck will be the major factor determining a single result.
To know whether this model was reliable and accurate, we would need to use it over hundreds,
indeed probably thousands of matches, yes, really thousands. Later in the book, I will show you
why.

What about our quest to predict the finishing league positions? We can run the Dixon-
Coles model in the same way for all the other remaining 91 fixtures, calculating the match
outcome probabilities of each one. With a total of 92 fixtures and 3 possible outcomes for each,
however, the number of potential different scenarios for the league table is huge, in
the power 92, written 3°, or 78,551,672,112,789,411,833,022,577,315,290,546,06C




--- Page 3 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

in full. Calculating the precise probabilities of all these scenarios to find the most likely
outcome for the league table would be a thankless task, even for Poincaré’s infinitely
powerful mind. Instead, we can turn to the Monte Carlo simulation to do the job for us.
This involves a two-step process to simulate the goals scored for the remaining matches.
Firstly, we will use a uniformly distributed random number using Excel’s RAND() function
to determine the Poisson probability. Secondly, we will invert the POISSON function to find
the number of goals from this probability, given the expected goals for a team as estimated
by the Dixon-Coles model. In fact, Excel does not offer a function for the inverse of the
Poisson distribution. However, one is available via the Real Statistics Excel Resource Pack
(you can find it via Google), which can be downloaded and installed as an Excel Add-
in (and switched on via Developer > Excel Add-ins from the Excel ribbon). The function
takes the form POISSON_INV(probability,mean), and calculates the smallest integer x such
that POISSON(x,mean,TRUE) > probability. Thus, by substituting RAND() for the probability
of scoring x goals, POISSON_INV(RAND(),mean) will output a random goal tally for a team
where ‘mean’ is the expected goals (from the Dixon-Coles model). For example, if RAND() =
0.27 for Aston Villa against Sheffield United, POISSON_INV(0.27,1.264) = 0 goals. Alternatively,
If RAND(Q) = 0.97, =POISSON_INV(0.97,1.264) = 4 goals. We can then build a Monte Carlo
simulation following the methodology described in the previous chapter to produce a large
number of random goal tallies for both teams for all remaining 92 matches, calculating league
points for each team based on the match score. Finally, these Monte Carlo iterations allow us to
calculate the expected finishing league table. My Monte Carlo simulation consisted of 10,000
iterations, and the expected finishing points for the 20 teams (the arithmetic mean of the
10,000 simulated points totals) was as follows.

1 minute left in chapter

Actual
Position

Points
Difference

L
Manchester City

Leicester City

Sheffield United

Tottenham

Arsenal

Everto

San 7 seen en Write A icleicneneeeanen Oetrinoneae

Newcastle United

Watford

West Ham United
Bournemouth
Aston Villa

Norwich City

I’ve also included the actual finishing points, actual finishing positions and the difference
between actual and expected points (a negative value denotes an underachievement with
respect to model expectation). Broadly speaking, the Monte Carlo simulation predicted
the finishing points of teams reasonably well; 12 of the teams were within +1 win/
loss of expectation. However, there are some notable exceptions. Norwich City, Crystal
Palace, Leicester City and Watford substantially underperformed in the final 9 games of
the season. It cost Leicester City a Champions League spot and Watford relegation to the
Championship. Meanwhile, Southampton and Tottenham substantially overperformed relative
to model prediction. These deviations from expectation could be random. Alternatively, they
might have some underlying causal explanations. Perhaps poor performers might blame the
lack of crowds, and in the case of Norwich might have given up trying after they were
relegated with 3 games still to play. Better performers, meanwhile, might have undergone

15%




--- Page 4 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

structural improvements to their game. In the case of Southampton, the humiliation at the
hands of Leicester in October 2019 may have heralded a turning point. Both Tottenham
and Southampton continued to perform well, and over games played since the restart of
the 2019/20 campaign in June 2020 and into the 2020/21 season, were ranked 1st and 5th
respectively on 26 November 2020 when I ran the model.

We can test to see if any of the deviations between expectation and actual outcome are
statistically significant. Cast your mind back to the primer on statistics, and in particular my
discussion of statistical hypothesis testing. Our null hypothesis would be that an actual team
performance was either lucky or unlucky relative to expectation. The alternative hypothesis
would be that, in fact, something else caused the deviation. Of course, there is a third
hypothesis: my model is wrong, but we won't worry about that for now. How likely is it that
Norwich, for example, will collect just 21 points, 7.64 fewer than the model predicted? We can
count the number of times it happened in the Monte Carlo simulation, and hence calculate the
percentage (and implied likelihood) as a fraction of the total number of iterations. It happened
only 74 times, implying a probability of 0.74%. This frequency, and those of all the other points
totals that were observed can be plotted in a frequency distribution.

Distribuion of Norwich's final Premier League points from a
10,000-iteration Monte Carlo simulation

14%

12%

10%
ay 8%
2
= 6%
v
4%

O%

21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43

Final points

In fact, 21 points is the fewest possible, since this was the points total Norwich held when
the Premier League was suspended. After resumption, they lost their 9 remaining fixtures.
With an implied probability of less than 1% that Norwich would do so the weaker thresholds
of statistical significance (5% and 1%) have been met. Hence, we might conclude that the
null hypothesis has been disproved, in favour of the alternative one, where something other
than chance may have accounted for their terrible performance. Of course, we should remind

1 minute left in chapter

ourselves that this sort of conclusion is simply subjective and based on arbitrary conventions
about statistical significance. Given that my Monte Carlo simulation can produce 74 occasions
in 10,000 where Norwich lose all 9 games completely randomly, we should always remain
open to the view that chance was the only thing at play. This sort of inference testing doesn’t
prove that something caused Norwich to lose all 9 (although that may well be the case); it
simply informs you about the chance of seeing something, given nothing was causing it at all.
Remember: with enough iterations, anything is possible by chance.

We could also directly calculate the probability of Norwich losing all the games from the
match odds themselves. After removing the bookmaker’s margin (later, I will show you how),
I calculated the probability of Norwich losing each game. Then, using the multiplication rule,
I multiplied these all together to calculate the probability of them losing all 9. The answer was
0.38%. That’s within the same ballpark as the figure generated by the Monte Carlo simulation.
For the record, the most points Norwich could have finished with is 48 (from 9 wins). The
highest total they achieved in 10,000 tallies was 43. This is hardly surprising. Based on the
actual match odds, the probability of 9 wins was roughly 1 in 60 million; we’d have needed 60
million Monte Carlo iterations to have a reasonable chance of notching up just one 48-point
total.

How often did Manchester City, Manchester United, Chelsea, Leicester City and others finish
in Champions League spots? My Monte Carlo simulation can answer that too. Unsurprisingly,
Liverpool occupied 1st place in all 10,000 iterations. They were, after all, 25 points ahead when
the League was suspended, and Manchester City only had a maximum of 30 points available to
win. Manchester City, however, were never outside the top 4, implying they were effectively a
certainty to achieve a Champions League place, although a larger Monte Carlo simulation may
well have witnessed a few iterations where that would happen. For the other hopefuls, their
implied top 4 probabilities were as follows.




--- Page 5 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

% top 4 Bookmakers

Leicester
Manchester United
Sheffield United

Tottenham

Arsenal

Everton

The remaining 8 Premier League teams failed to register a single top 4 finish in any of the
10,000 iterations. Leicester City actually finished 5th, yet the implied odds from this model
for them winning a Champions League place were just 1.10. They were the big losers, with
Manchester United, an outsider before the League resumed, taking their place. How do these
implied probabilities compare to the bookmakers’ odds that were available at the time? I’ve
shown prices that were available just before the resumption of the League in June in the final
column. Remember, when looking for a good bet, we aren't interested in finding winners —
although it goes without saying that it’s nice to win - we are interested in finding value. And
recall, expected value is any bet where the odds the bookmakers are offering are longer than the
‘true’ odds. If my Monte Carlo model was accurately predicting the ‘true’ top 4 odds, then you
can see that there was value available in bets on both Leicester City (1.14/1.10 - 1 = 3.6%) and
Manchester United (2.75/2.70 — 1 = 1.9%). One of them turned out to be a winner, the other
a loser. To reiterate, two bets can tell us nothing about the validity of the prediction model;
for this we need many hundreds or thousands more. Incidentally, Manchester City to finish in
the top 4 were available at 1.02. Given there were no occasions in 10,000 iterations where they
finished outside, these odds would also appear to represent value. (1.02/1.00- 1 = 2%).

You may have also noticed that for the less likely top 4 finishers, or longshots - Wolves,
Sheffield United, Tottenham, Arsenal and Everton - there is an ever-increasing divergence
between their model-implied ‘true’ odds and the bookmakers’ odds, with the latter being
much shorter. For the favourites, they are much closer. Everton, for example, would have been
predicted by my Monte Carlo simulation to make top 4 less than 1 time in a 1,000. Yet, the
bookmaker was only prepared to lay 26. Given the probability that Everton had of making
top 4, why wasn’t the bookmaker prepared to offer me much more, should I win? There are

1 minute left in chapter

a number of explanations for this, which readers of my previous books may be familiar with.
Some of these relate to the psychology of bettors, in particular their miscalculation of low
probability outcomes — humans exhibit a bias towards overestimating their likelihood - and
their exploitation by bookmakers. Others relate to the greater impact of uncertainty for lower
probability outcomes, and the potentially greater liabilities they create for the bookmaker. If
your model is wrong by 1% about Manchester City, that’s the difference between ‘true’ odds
of 1.00 (100%) and 1.01 (99%). By contrast, if it’s wrong about Everton, that’s the difference
between odds of 1,429 (0.07%) and 93 (1.07%). Since bookmakers are also using models to
estimate the ‘true’ odds, there’s obviously a much bigger scope for a pricing error, the longer the
odds are. Far safer, then, to err on the side of caution and quote odds that are much shorter than
they should be. In terms of what the bookmaker is doing, this means that when they create a
margin (or vig) they are not distributing it equally across all possible outcomes but are actually
placing more of its weight, or emphasis, on the longshots than the favourites when those are
present in their betting market.

The disproportionately bigger deviation between the ‘true’ odds and the bookmakers’ odds
as they lengthen is known as the favourite-longshot bias. It can be found in many sports
betting markets, including football, tennis and horse racing. Since longshots, compared to
favourites, are priced disproportionately shorter than the ‘true’ chances imply, it means that
you can expect to lose disproportionately more per stake betting on them blindly. That is to
say, the expected payout (or return) will be less than for favourites. For example, in the home-
draw-away football betting markets of the major European football leagues, betting a £1 stake
on the average available bookmaker’s odds on a team when they were longer than 20, between
the seasons 2012/13 to 2019/20, would have lost you an average of £0.37, or more than a third.
By contrast, if you’d have bet the odds that were shorter than 1.33, you'd have actually made
a little bit of money without doing any predicting at all. Free money, you might think. Well,
yes, in this case it would have been, but here some of that could have been good fortune. More
typically, we would still expect losses on the shortest-priced favourites, but much smaller than
for the longshots.




--- Page 6 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

The Favourite—Longshot Bias in European Football Match
Betting Odds

5%

0%
vo
Ss -5%
74
zs -10%
= -15%
3}
a -20%
2 -25%
7) >
op -30%
B 35%
>
a
~ -A0%

ols als o\e o\s ols os alo ols Ny ole oo
P ie 2 © eo Ky © SP P 7
9! i) o

Implied bet win probability

Despite the less unfavourable odds this bias can offer for favourites, bettors continue to love
betting on the longshots, and bookmakers continue to worry about the liabilities this creates
for them, so the favourite-longshot bias persists. It is an example of what economists call
a market inefficiency, when there is a structural and non-random way in which the quoted
value of things (in this case the bookmakers’ odds) does not reflect accurately the underlying
fundamental value (the ‘true’ chances of football teams winning).

What about the odds for relegation? 8 Premier League teams, shown in the table below, had
at least one Monte Carlo iteration where they were relegated.

1 minute left in chapter

Bookmakers’

Implied

| % relegated

Norwich

Aston Villa

| Bournemouth
L.

| West Ha

| Watford

Most likely was Norwich. As we’ve already seen, they were indeed relegated, losing all 9 of their
remaining games. The Monte Carlo simulation gave them less than a 3% chance of avoiding
relegation. You could have had odds of 1.10 at one bookmaker. That would have been a steal
given the short odds (1.10/1.03 - 1 = 6.8% expected value). The odds of 1.40 for Aston Villa also
offered expected value (1.40/1.35 - 1 = 3.7%), but that would have proved to be a losing bet.
To reiterate, however, one unexpected outcome does not invalidate a model, any more than one
expected one validates it. Once again, you will notice the unfavourable pricing of the outsiders
- or longshots - for relegation.

Although a small number of results cannot validate or invalidate a prediction model like this
Monte Carlo simulation, all bettors who use them should remain ever-alert to the possibility
of flaws. I’ll say it again: the outputs of a model are only as good as the inputs that go into
it. What might some of the possible flaws in this model be? An obvious one was failing to
update the attack and defence strengths of teams as each round of fixtures was played. Doing
so would arguably make them more relevant in terms of the most recent performance. Perhaps
more generally it might be considered inappropriate to use such a large sample of fixtures to
calculate attack and defence strengths, going back to the beginning of the season. Dixon and
Coles attempted to solve this problem by applying a stronger weighting for more recent games.
To save time, I just decided not to bother.

Another potential source of error was the disappearance of crowds; in particular, how would
home teams play without their fans? In fact, once major European leagues which restarted had
completed their 2019/20 season, there was shown to be a weak but obvious drop in home win
percentage of about 2%. In the following season, with fans still absent, the size of this decrease
grew in some divisions. In the Premier League, for example, there were in fact more away wins
(153) than home wins (144) in the 2020/21 campaign, something previously unheard of. Of
course, if home teams are winning less and away teams winning more, that shouldn’t have

17%




--- Page 7 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

much impact on final league points. What a team might lose in performance at home, they will
gain away. Nevertheless, there were other reported changes, in particular, how the game was
played and how the referee controlled it. These might have had an influence on the goal scoring,
and hence the number of expected goals. Since the Poisson model which lies at the heart of
my Monte Carlo simulation relied on expected goals, we shouldn’t underestimate the possible
influence that playing without crowds may have.

Perhaps the most significant potential model error lies in the assumption that team goals
can be Poisson-distributed in the first place. For this to be true, goal scoring should be
independent, but is that really the case? Arguably, teams which fall behind may become more
motivated than they were to redress the balance. On the other hand, teams that draw level
might be more motivated to push on. Whatever the dynamics at play, it surely cannot be denied
that goal scoring in football will involve a considerable degree of player and team psychology.
If that is the case, the idea that goals just go in at random must surely be brought into question.

Being ‘well-documented’ means many bettors are potentially already using the Dixon-
Coles model for football match prediction. The corollary to this is the implication that the
odds bookmakers quote will already reflect the information it brings. A statistical model
about a football match is simply a means of quantifying information about the teams, when
attempting to find the match outcome probabilities. The better the information, and the better
the way it is modelled, the more valid (reliable and accurate) the probabilities will be. The
bookmakers’ odds are simply the public expression of their model and the information that
went into it. They may change the odds if they receive new information that requires their
model to be updated. That could be in the form of news about the teams, or it could be in
the form of money bet by their customers, some of whom may also be using different, and
possibly superior models. If a lot of money comes for one of the sides, the bookmakers may
start to believe their initial model was wrong. The shortening of their odds (whilst lengthening
the odds for the other team) is then a measure of the correction to their model. Odds may
move many times to reflect the continual flow of new information about the match. As the
total amount of information about it increases, arguably the accuracy (or what economists
call efficiency) of the odds improves. This process is called price discovery, whose consequence
typically, although not always, is to arrive at odds that most accurately (or efficiently) reflect
the ‘true’ match outcome probabilities.

Being, as they are, a reflection of a collection or crowd of opinions, the odds might be
regarded as an expression of the ‘wisdom of a crowd’. I’ve discussed at length this concept in
my last book, Squares and Sharps, Suckers and Sharks: the Science, Psychology and Philosophy of
Gambling. Indeed, later I will show you how you can use it to make a sustainable profit. Some
have argued that, really, most bettors are pretty dumb, backing teams on a whim, with the odds
instead reflecting the wisdom of the bookmakers’ odds compilers and a few other very savvy
customers. Be that as it may, even if this premise is valid, it doesn’t really invalidate the concept;
it just reduces the size of the crowd. The significant take-home message, however, is this: if the

1 minute left in chapter

model you are using is already reflected in the odds, because other people are using it too, it
won't be good enough to help you overcome the bookmakers’ margins. If you want to beat the
odds and find for yourself some expected value in what the bookmakers are offering, you will
need to have a prediction model that is better than theirs, and arguably also better than a whole
load of their customers, too, who influence the bookmakers’ odds.

Some statisticians have advocated abandoning Poisson and using other distributions to
model football goals instead, for example the Weibull distribution and the negative binomial
distribution, but I have no plans to go there in this book. Instead, I think it’s time to move on to
another Monte Carlo simulation.

The unlikeliness of Leicester City’s 2015/16 Premier League title

No one, not least the supporters of Leicester City, will forget the team’s remarkable winning
of the 2015/16 Premier League. At a time when the big 6 teams - Chelsea, Manchester
City, Manchester United, Arsenal, Liverpool and Tottenham - were beginning to establish
themselves as a class above the rest, Leicester City, under the management of Claudio Ranieri,
performed the unthinkable and finished ahead of them all on 81 points. Admittedly, this was
one of the lowest winning totals of the Premier League in a season when all the regular big
players appeared to have taken leave of absence, but the scale of the achievement cannot be
underestimated. Bookmakers had offered 5,001 (or 5,000/1 in fractional format) for them to
win at the start of the season. They no longer offer such long odds on any team, even though
such a price was already hugely discounted via the favourite-longshot bias. What exactly were
the ‘true’ odds of Leicester City winning the Premier League that season? One rudimentary
way to estimate that would be to measure how likely it was they would have won the Premier
League the season before and then assume their level of performance would broadly remain the
same.

I could use something like the Dixon-Coles model I introduced earlier in this chapter to
estimate the chance of Premiership teams winning their matches. I could make it more reliable
by updating the attack and defence strengths after every round of fixtures, to reflect each
team’s ability as it waxes and wanes through the course of the season. Arguably, however, a
much easier method, for this exercise at least, would be to just borrow the model that the
bookmakers use to create their football match odds for every game; that is, use their odds
directly as a measure of the implied probabilities. After all, bookmakers will be using similar
models to Dixon-Coles to produce their odds; why try to reinvent the wheel when one has
already been invented for me? What is more, their model is likely to be better than mine
at estimating the ‘true’ match odds, because that is their job and they have access to more
sophisticated data and data analysis tools than I do. Regardless, even if I did have a better
match prediction model than theirs, capable of making profits from betting on the matches,

17%




--- Page 8 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

it arguably won’t make a huge amount of difference to the outcome here, which is to say how
likely Leicester City were to win the 2014/15 Premier League the season before their triumph.

A bookmaker’s match odds, in a way, are to match outcomes what xG are to goals; they
strip out the luck. Whilst xG tells us how many goals we would have expected to see, given
the quality and quantity of the scoring chances in a game, the match odds tell us the expected
outcome in probabilistic terms, albeit before the match kicks off. At the end of a 90-minute
football match, there are three possible outcomes: home win, draw or away win. If you were to
assign numbers to these outcomes, you might use 1 for the outcome that happened and 0 for
those that didn’t. Prior to the match, however, we can use the odds to express the probability
of each of these three results occurring. Remember, just calculate their reciprocal. Suppose the
home odds were 2, the draw odds were 3 and the away odds 6. The home probability would be
0.5 (or 50%), the draw 0.333 (or 33.3%) and the away 0.167 (or 16.7%). We can then say that
the expected amount of home win was a half, the expected amount of draw was a third and
the expected amount of away win was a sixth. The actual result is just a consequence of the
hidden random variables, in the same way that actual goals are. The outcome might deviate
substantially from expectation, but that does not mean the former invalidates the latter,
certainly not over just one game. The away team might win, but it is still perfectly acceptable to
say they had a sixth of an expected win.

What if we now factor in League points for winning (3), losing (0) and drawing (1)? If the
home team had a 50% chance of winning, and 33.3% chance of drawing and a 16.7% chance
of losing, then their expected points can be calculated by multiplying each probability by their
respective points and summing the products. Thus, the home team had (0.5 x 3pts) + (0.333 x
1pts) + (0.167 x Opts) = 1.833 expected points (xP). Similarly, the away team had (0.5 x Opts) +
(0.333 x 1pt) + (0.167 x 3pts) = 0.833 xP. Of course, the game will either finish with 3 points
awarded to one of the teams and 0 points to the other, or 1 point to each, but xP tells us the
average number of points we should expect each team to win if the exact same game could be
played an infinite number of times. Since betting odds are available for all 380 matches in a
Premier League season, we can simply calculate the xP in this way and sum the totals for each
of the 20 teams over their 38 games in a season. This will give us their expected finishing points
totals, and hence their expected League positions.

We already know, however, that bookmakers’ odds are not fair and the implied probabilities
of the three possible outcomes sum to more than 100%. Thus, we must remove their margin
before we calculate the fair outcome probabilities and the xP. To do this, we need to know how
the bookmaker applied their margin in the first place. Unsurprisingly, they won’t tell us, so we
have to guess. A rudimentary method would apply it evenly across every possible outcome,
but we also now know that this is not the case. There is a bias to the application of this
margin, a favourite-longshot bias, where a bigger margin weight is applied to longshots than
to favourites. Through my website Football-Data.co.uk, I have discussed at length some of the
possible ways that this margin could be applied, and hence removed by reverse engineering.

1 minute left in chapter

These include my own method that assumes the size of the margin weight is proportional
to the odds (double the odds, double the margin weight), and a couple of others: the odds
ratio and a logarithmic function. You can find detailed mathematical descriptions of these
methodologies via my website (see the link ‘Wisdom of Crowds’), in addition to an Excel ‘Fair
Odds’ calculator that will do the margin stripping for you. There is also work in this area from
the financial economist Hyun-Song Shin who was one of the earliest theorists to discuss the
favourite-longshot bias and how to account for it when removing the bookmakers’ margins,
but his methodology is arguably better suited to horse racing, which typically has more
runners and thus more possible outcomes.

My preferred method for football home-draw-away betting odds is the logarithmic function.
Briefly, this assumes that the bookmakers transform their fair win probabilities for each
outcome into unfair ones by means of a common exponent. An exponent is simply a number
representing the power to which another number is raised, for example 3 in 2? (= 2x2 x2 = 8).
Since probabilities are less than 1, raising their power with an exponent greater than 1 would
reduce them. For example, 0.51.05 = 0.484. Bookmakers want to increase these probabilities to
build an overround and a margin. Thus, the exponent must always be less than 1. 0.59.95 =
0.518. Similarly, 0.3339-95 = 0.352 and 0.1679-95 = 0.182. Summing these unfair probabilities
makes 1.052 or 105.2%. The corresponding odds are now 1.93 (home), 2.84 (draw) and 5.49
(away). You can see the longer odds have had a proportionally bigger margin weight applied to
them. The actual weights are 1.035 for home (2 / 1.93), 1.056 for draw (3 / 2.84) and 1.094
for away (6 / 5.49). Simple but effective; it’s called a logarithmic function, but it might just as
well be called a power function, since a logarithm is just the opposite of a power. If 2? = 8, then
logs8 = 3, where the number 2 is known as the base. Here, you would say log to the base 2 of
8 = 3. Similarly, if 0.59.95 = 0.518, then logy ,0.518 = 0.95. Likewise, logy 3330.352 = 0.95 and
logy 170.182 = 0.95. To reverse engineer the application of the margin in this way we need to
use logs, so perhaps that’s why the name is preferred.

We should also decide which bookmaker we are going to use for this exercise. Since we want
as accurate an estimate as possible of match outcome probabilities, it would make sense to use
the bookmaker that quotes the most accurate (or efficient) odds. But how to tell? Over many
years of odds data analysis, I have found that the best bookmaker for this task is Pinnacle.com.
I will explore the reasons for this in more detail later in the book; for now, it is enough to say
that their business model demands it. Most other bookmakers use a different business model
to theirs, and hence quote prices slightly differently and arguably less accurately. If that is so,
I hear you wonder, is it easier to find expected value from their odds? Yes, it is. Again, I will be
looking at how we find it, and what the implications of exploiting it are.

Furthermore, which odds should we use? Pinnacle’s first quoted odds, called the opening
odds, based solely on the interpretation of their match prediction model? Or odds that have
been updated after more information and money has been received? The time at which the
most amount of information is available and the most money wagered on a game occurs just

18%




--- Page 9 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

before the kick-off. These are called the closing odds, because kick-off is the time the betting
market will close, unless the bookmaker has an option to put it into play, with odds updating
as the match progresses. Arguably, the closing odds are the most accurate of all odds, at least on
average. Undeniably, they are more accurate than the opening ones; I have carried out further
data analysis to demonstrate this to be true. Again, more on that later.

Having removed the margin from Pinnacle’s closing odds (for this exercise I used the
logarithmic function), I calculated the xP for the 2014/15 Premier League season. Since we
are using the match odds as the season progresses to calculate the final season xP total, in
effect we are gauging how the bookmaker updated their opinion about the relative prospects
of each team. A statistical method that assigns probabilities or expectations based on current
information prior to observation, which are then revised post observation after new data has
been collected, is known as the method of Bayesian inference. For example, a bookmaker will
have an opinion about Liverpool defending the Premier League title and assign a probability to
that. After being thrashed by Aston Villa 7-2, it is likely that the bookmaker will want to update
the probability given the new data, most likely reducing it because they no longer believe them
to be as defensively good as previously thought. A similar downgrading of Liverpool’s chances
occurred when Virgil van Dijk, their key central defender, was ruled out for the remainder of the
2020/21 season following injury in the Merseyside derby against Everton on 17 October 2020.

Using the method described above for calculating xP, Leicester City amassed a total of 43.78,
placing them 13th in the xP table (they actually finished 14th with 41 points). Teams finishing
so low in the table and with so few points (just 6 ahead of relegation), don’t win the following
season’s Premier League, do they, particularly when they sack their manager (Nigel Pearson)
who’d engineered the team’s Premier League survival, have seen their best player (Esteban
Cambiasso) leave, and acquired the services of a new manager (Claudio Ranieri) who’d never
won a league title and had only recently been sacked as an international manager (Greece) for
losing to the Faroe Isles. Thus, it would be a reasonable assumption that at best, Leicester City
could be expected to achieve much the same in 2015/16 as they had in 2014/15: that is, survive
relegation. Even when they sat in 5th place in October 2015, Stats.com’s supercomputer was
predicting a 13th place finish with 45 points.

xP can tell us where we think Leicester City will finish in the 2015/16 Premier League
table, but it can’t tell us the likelihood of them winning the League, or indeed the chances
of finishing in any of the 20 possible positions. For that we need a Monte Carlo distribution,
since calculating the probabilities of all possible scenarios for all 20 teams mathematically is
impossible. It was bad enough for 92 games. Now we have 380, and 338° possible outcome
scenarios. I won't write this number out in full - it has 182 digits! Using a random number to
simulate a match outcome in the same way I did for the last Monte Carlo exercise, I simulated
the entire 2014/15 Premier League season 100,000 times. The mean xP for all teams matched
almost exactly those calculated using the xP formula. The worst match was still within just
0.2% and the majority were within 0.05%. Leicester again amassed a mean of 43.78 xP and

1 minute left in chapter

finished in a virtual 13th place. This time, however, we can see the distribution of possibilities.
I’ve shown these for Leicester City below. The distribution, although discrete - it must be since
League points are discrete - takes on the familiar shape of the continuous bell-shaped normal

distribution.
Distribution of Leicester City's virtual 2014/15 Premier
League points via a Monte Carlo simulation
6%
5%
> 4° 0
ay
io) 3%
2 2%

15 18 21 24 27 30 33 36 39 42 45 48 51 54 57 60 63 66 69 72 75 78

Final League Points

We can test if the normal distribution offers a reliable approximation by seeing if the data
conform to the empirical rule, where 68.2%, 95.5% and 99.7% of the data fall within 1, 2 and 3
standard deviations, respectively. Here, the figures are 69.5%, 95.3% and 99.8%. This is hardly
surprising since the results are simulated randomly. A normal distribution is always the telltale
sign that the system under investigation is random in nature. The minimum and maximum
tallies are 15 and 79 points. Indeed, 79 was an outlier; the next highest was 75. The standard
deviation (recall, that is how far, on average, each score lies from the arithmetic mean) is 7.29
points. Leicester City, as we know, finished the 2014/15 season with 41 points, much less
than 1 standard deviation from the mean xP and well within the most probable region of the
distribution.

We can repeat this exercise for League positions, as the next chart illustrates.




--- Page 10 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

Distribution of Leicester City's virtual 2014/15 Premier
League positions via a Monte Carlo simulation

Frequency

123 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

Final League Points

Their best finishing position was 2d —- it happened 3 times in 100,000 Monte Carlo iterations.
They finished 3:4 a further 19 times. The probability of relegation was 13.14%. The average
finishing position was 13th rounded to the nearest integer; one place higher than their actual
finishing spot.

Let’s review our working hypothesis: the performance of Leicester City in 2014/15 will be
repeated in 2015/16. By performance, we actually mean ‘expected performance’ as predicted
by this Monte Carlo model, using the 2014/15 match odds as the inputs. Hence, we would
conclude that the chances of Leicester City winning the 2015/16 season are less than 1 in
100,000. When I first ran the model a few years ago with different odds and a different method
for removing the margin, my Monte Carlo simulation had one iteration in the 100,000 where
they became champions. Broadly speaking, however, my calculated xP and expected rankings
are the same second time round. Thus, the predicted odds should be no shorter than 100,000.
Furthermore, in 100,000 model iterations of the 2014/15 season, the team never amassed
more than 79 points. Yet in 2015/16 they finished as champions with 2 more, a whopping 5
standard deviations above expectation.

We should already suspect that the probability of achieving 81 points based on this
hypothesis will be less than 1 in 100,000. Since we have shown that the spread of virtual points
in the Monte Carlo simulation can be approximated by the normal distribution, Excel provides
a useful function to directly calculate this: NORM.DIST(x,u,0,cumulative). The function’s
parameters are described in the parentheses. x is the required value for which you want to
calculate the probability, u is the arithmetic mean of the distribution, o is the standard
deviation, and ‘cumulative’, as for the POISSON Excel function, is a logical argument that
determines the form of the function. If cumulative is TRUE, NORM.DIST returns the
cumulative distribution function or the total area in the distribution to the left of x; if FALSE, it

1 minute left in chapter

returns the probability density function or the probability density at x. Being a continuous
distribution, the interpretation of ‘probability density’ can be a little hard to conceptualise for
those not familiar with integral calculus. Fortunately, we need ‘cumulative’ to be TRUE, which
is far easier to comprehend. With Leicester City’s xP of 43.78 and a standard deviation of 7.29
points, NORM.DIST(81,43.78,7.29,TRUE) returns an answer of 0.999999835. This can be
interpreted as meaning that 99.9999835% of the points totals should fall below exactly 81, and
0.0000165% above. That’s approximately 1 in 6 million. Of course, since virtual points in a
Monte Carlo simulation are discrete and can only be integers, with no possibility of an infinite
number of decimal points in between them, the use of a continuous data function isn’t wholly
appropriate, but it does at least give you a ballpark idea of how unlikely 81 points actually was,
based on the expectation that Leicester City would perform much as they had the previous
season. Excel does have a binomial function, BINOM.DIST(s,t,p,cumulative), for discrete data
(where s is the number of successes in a series of independent trials, t, each with success
probability p) but there are clearly practical obstacles to using it here. Firstly, with three
possible match outcomes, we don’t have a binomial proposition. We could attempt to reduce
Leicester City’s performances to ‘wins’ versus ‘not wins’, but each match, furthermore, has a
different success probability. To use this function, the probability of success on each trial must
be constant. Naturally, in situations such as these, where calculating exact probabilities and
probability distributions is difficult, this is precisely the reason why we turn to the Monte Carlo
simulation in the first place, to estimate them.

Ire-ran the Monte Carlo simulation using the 2015/16 match odds to see what Pinnacle.com
had been making of Leicester City’s relative prospects compared to other teams as the season
progressed. Obviously, they did not fall away as the Stats.com supercomputer had predicted;
they continued to win, often by small margins and by playing counter attacking football.
Consequently, Pinnacle started to reflect some of this elevated performance in the betting odds
for later matches, making them a little shorter than they would have been otherwise. Evidently,
however, they still regarded much of what Leicester City were doing as just lucky. By the end
of the season when they were champions, their expected position was 9th with 53.13 xP, just
9.35 more than the previous season and still nearly 28 points (and the best part of 4 standard
deviations) behind their actual total of 81 points. 177 of the 100,000 iterations saw them
crowned champions, about one in 565. In just 20 iterations (or 0.02%) were they able to equal
or better 81 points, the maximum being 86. With a standard deviation of 7.66, the function
NORM.DIST(81,53.13,7.66,TRUE) returns a figure of 99.986%, meaning 0.014% (or about 1 in
7,000) would be expected above 81 points, a close match for the actual proportion returned by
the Monte Carlo simulation. Even when judged by the match odds of 2015/16, let alone those
of the previous season, Leicester City had massively over-achieved relative to expectation. No
other team has come anywhere close to this level of deviation from expected performance,
before or since.

One question remains: was this over achievement just pure chance, a once-in-a-thousand

19%




--- Page 11 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

blue moons, or is there something wrong with the Monte Carlo model? After some reflection,
I think the answer is a bit of both. Yes, Leicester City had been incredibly lucky, but arguably
not so lucky to the tune of less than 1-in-100,000. A couple of reasons might be invoked to
argue that the model is flawed. Firstly, the team’s collective ability may have improved to a
level that was not sufficiently being reflected in the match odds during the 2015/16 season. As
such, the 2014/15 odds would not provide a reliable prediction of what to expect during the
following season. And Pinnacle (along with other bookmakers) were too slow during 2015/16
to fully notice. Of course, we can only learn these things with hindsight, and hindsight is
never a great way to invalidate a prediction model. Perhaps more importantly, using a random
number to simulate match wins and losses may not be appropriate. Doing so implies that
match results are independent, where the next one has no memory of the previous. That
might be true of roulette balls but arguably it’s not true of Leicester City’s players, who
all have memories capable of expressing emotions about match outcomes and translating
those into performances for subsequent games. Consequently, possible League points may not
normally distribute at all. The invalidation of independence might also account for some of
the explanation, in addition to the favourite-longshot bias, for the much shorter odds for top
4 placing and relegation the bookmakers had offered compared to the odds implied by my
previous Monte Carlo simulation for the 2019/20 Premier League season.

Whatever distribution League points do take, the tails could be fatter than the normal one,
meaning rare events are actually a lot more likely than the normal distribution would imply.
However, because the precise shape of the distribution is unknown, we cannot calculate the
mean and standard deviations, and hence we are unable to calculate the true probabilities of
these rare events. All we can say is that they are probably more likely than we thought they
were. Was this event more likely than the odds of 5,001 implied? It’s a moot point. Such events
have been coined by the risk analyst Nassim Taleb as ‘Black Swans’ (a term that became the
title of his bestselling book); they lie beyond the realm of normal expectations and probability
theory. In that regard, at least, Leicester City winning the Premier League can perhaps be
reasonably described as a Black Swan.

A Monte Carlo Simulation for Tennis

Suppose there is a tennis match where player A plays a single game against player B. The
complementary rule tells us that if the probability that player A will win a single point is p,
then the probability that player B will win a point is 1 - p. What is the probability that player
A will win the game? The scoring system in tennis goes 15, 30, 40, then game. However, at a
score of 40-40, or deuce, the winner of the next point gains an ‘advantage’. If this player wins
again, they win the game, but if they lose the score returns deuce. Like the penalty shootout,
this is an example of what is known, in statistics, as a Markov chain (named after the Russian

1 minute left in chapter

mathematician Andrey Markov), a sequence of random and independent events in which the
probability of any particular sequence outcome depends only on the probabilities of each and
every step in the sequence.

Calculating the probabilities of all the different ways player A can win the game will arguably
present a challenge. Following the multiplication rule, player A can win to love (where their
opponent fails to score) by winning the first 4 points with a probability p+. Alternatively, they
can win after 40-15 with probability 4p4(p— 1). This is derived from multiplying the probability
of winning 4 points (p+) by the probability of conceding 1 point (1 - p). There are then 4 possible
ways to concede 1 point and still win the game to 15 (at love, at 15, at 30 and at 40). Hence,
we multiply the product p4(1 - p) by 4. They could also win to 30. There are 10 possible ways
to win 4 points and concede 2 so the probability is 10p4(1 — p)?. Finally, there are 20 possible
ways to reach deuce, where each player wins 3 points. Hence, the probability is 20p3(1 — p)?.
However, how do we calculate the probability of a win from here, given that the nature of the
rules implies that the game could, in theory, go on forever? For those far more mathematically
minded than I am, it is possible to determine this via an infinite series. I had to Google it. The
probability, it turns out, is p? / (1 - 2p(1 - p)). Thus, the probability of winning via deuce is
20p3(1 - p)? x p? /(1- 2p(1-p)) = 20p*(1 — p)? / (1- 2p(1 - p)). Thus, following the addition rule,
the probability that player A wins the game is p4 + 4p4(p— 1) + 10p4(1 - p)? + 20p°(1-p)? /(1-
2p(1-p)).

As with the penalty shootout, deuce is about the point where I typically give up with
mathematical first principles. We are still only in the first game. To calculate the probability of
a player winning a set, we then need to repeat this exercise all over again, factoring in the rule
for winning by 2 clear games, or by tie-break if the set goes to 6-6 (which must be won by 2 clear
points). We must then repeat this process for calculating the probability of a player winning the
required number of sets to win the match, before their opponent manages it. By the end, the
equation is going to be a monster. Let’s abandon the maths and build a Monte Carlo simulation
instead. The process is much the same as for the penalty shootout. For the simplest of models,
there are only two significant inputs: the probability of player A winning a point when serving
and the probability of player B winning a point when serving. We don’t need to worry about the
probabilities of winning a point when receiving, since this is the complement of the probability
of the server winning. It’s then simply a matter of using the Excel formulae to build in the
rules of tennis point scoring and simulate the evolution of a match a large number of times. My
Monte Carlo simulation had 10,000 iterations to calculate the probability of one player winning
a best-of-3 sets match with a final set tie-break. The chart below shows how the probability
of winning the match varies as a function of the probability of player A or player B winning a
service point.




--- Page 12 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

Probability of winning a 3-set tennis match as a function of

the probability of winning individual service points
100%
90%

80% — Player B 50%

Player B 55%

70%

>
5

2 60% Player B 60%
a Player B 65%
BM Player B 70°
‘5S ayer B 0
= ° ;

- 40%

2 30%

= 20%

10%
0% —

20% 30% 40% 50% 60% 70% 80%

Player A service point win percentage

You can see, unsurprisingly, that where both players are equally rated, they each have a
50% chance of winning the match. Essentially, the contest becomes a coin toss. When one of
the players has a relative marginal superiority in their service win percentage, the probability
of match victory increases quite dramatically. Just a 10% superiority in service point win
percentage translates in to an 85% to 90% probability of winning a 3-set match. This would
rise to about 90% to 95% for a 5-set match. This is the process of accumulative advantage,
where a small advantage one player holds over the other in each step is compounded into a
much bigger advantage over a series of iterative steps during the course of a match. Tennis
is an iteratively-played competition comprised of sometimes hundreds of points, so small
differences in player ability will translate into big difference in players’ match winning chances.
It is for this reason that tennis frequently throws up some very short-priced favourites (1.01
or even shorter) and some very long-priced underdogs. It also provides an explanation for why
the greater proportion of success, for example as measured by Grand Slam titles, is held by just
a tiny proportion of the players.

Colloquially, accumulative advantage is known as the Matthew Effect. The term was coined
by sociologist Robert Merton in 1968 when he described how the more eminent scientists in a
team tend to get the most credit for the team’s work, regardless of who did the work, and takes
its name from the Parable of the Talents in the biblical Gospel of Matthew. Similar aphorisms
you may have come across are ‘the rich get richer, and the poor get poorer’, attributed to
the English poet Percy Bysshe Shelley, ‘winner takes all’ and the ‘Pareto Principle’. Wherever
there is the opportunity to apply one’s advantage iteratively over many steps, such asymmetry
in outcomes will be inevitable. One obvious area where this can be expected to occur is in

1 minute left in chapter

the world of betting. You may be familiar with the expression ‘the house always wins’. This
is directly a consequence of the iterative nature of most gambling games. Spinning roulette
wheels is an obvious example but betting every weekend on your favourite football team is
too. The ‘house’ might only hold a 2% advantage over you in one step, but over 1,000 steps, its
influence is going to accumulate significantly.

We can also use the Monte Carlo simulation to estimate the influence of relative player
superiority on the chances of winning games and sets. Unsurprisingly, since a set and a game
have proportionally fewer steps (points) in the iteration, the accumulative advantage is smaller.
I’ve shown the curves for a game, set and match win probability as a function of player A’s
serving win percentage for the case where player B’s serving win percentage is 50%.

Probability of winning a game, set or match as a function of
the probability of winning individual service points

100%
90%

80%

70%

60%

50%
40%
30%
20%
10%
0° 0

Game
Set
— Match

Win probability

Player A service point win percentage

My Monte Carlo, furthermore, will also estimate the probability of any specific match score (in
sets) or set score (in games), the expected number of total games played in the match, and the
expected game superiority of the winner over the loser. This information can all be translated
into betting odds and compared to those offered by the bookmakers. Let’s consider an example.

By the time Nadal and Djokovic faced each other in the 2019 Rome Masters, a best-of-3 sets
outdoor clay surface match on 19 May of that year, they had gone head-to-head 22 times on
that surface in their ATP careers (not counting their Davis Cup meeting). Over those matches,
Nadal had a winning service point percentage of 61.28%; Djokovic’s was 58.48%. Plugging
those values into the Monte Carlo model and running 10,000 iterations, the following outputs
were obtained.

1) Nadal wins 63.53% (implied odds = 1/0.6353 = 1.574).




--- Page 13 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

Djokovic wins 36.47% (implied odds = 2.742).
2) Nadal wins first set 59.35% (implied odds = 1.685).
Djokovic wins first set 40.65% (implied odds = 2.460).
3) Mean total games played = 24.61.
Iterations < 24.5 games = 51.13% (implied odds = 1.956).
Iterations > 24.5 games = 48.87% (implied odds = 2.046).
4) Mean total games played in first set = 9.88.
Iterations < 9.5 games first set = 48.55% (implied odds = 2.060).
Iterations > 9.5 games first set = 51.45% (implied odds = 1.944).
5) Nadal wins 2-0 = 35.06% (implied odds = 2.852).
Nadal wins 2-1 = 28.47% (implied odds = 3.512).
Djokovic wins 2-0 = 16.59% (implied odds = 6.028).
Djokovic wins 2-1 = 19.88% (implied odds = 5.030).

On the day, bookmakers offered the following (best) odds.

1) Nadal to win 1.57; expected return = 1.57/1.574 = 99.7%.
Djokovic to win 2.50; expected return = 2.50/2.742 = 91.2%.

2) Nadal to win first set 1.65; expected return = 1.65/1.685 = 97.9%.
Djokovic to win first set 2.50; expected return = 2.50/2.46 = 101.6%.

3) Match under 24.5 games 1.74; expected return = 1.74/1.956 = 89.0%.
Match over 24.5 games 2.25; expected return = 2.25/2.046 = 110.0%.

4) First set under 9.5 games 2.20; expected return = 2.20/2.06 = 106.8%.
First set over 9.5 games 1.72; expected return = 1.72/1.944 = 88.5%.

5) Nadal wins 2-0, odds 2.40, expected return = 2.40/2.852 = 84.1%.
Nadal wins 2-1, odds 4.00, expected return — 4.00/3.512 = 113.9%.
Djokovic wins 2-0, odds 5.00, expected return = 5.00/6.028 = 83.0%.
Djokovic wins 2-1, odds 6.00, expected return - 6.00/5.030 = 119.3%.

I’ve highlighted the expected value bets in bold; there were 5 of them. Remember, for expected
value we are looking for expected returns (the unit stake included) to be greater than 100%.
Anything less will be a loss-making proposition in the long run, assuming our prediction model
is correct. The average expected return for these 5 bets is 110.3%, provided we bet the same size
stake for all of them. In the event, Nadal won the match 6-0, 4-6, 6-1, meaning only 2 of the 5
bets proved to be winners, the first set under 9.5 games at 2.20 and Nadal to win the match 2-1
at 4.00. Nevertheless, this would have been sufficient to return a small net profit of 1.20 units
for a 5-unit outlay, or a 124% return on investment. That’s more than the average expected
return, but it’s clear why; the biggest contribution to the profit was the bet on Nadal to win
2-1. This only had 28.47% chance of winning; the fact that it did means we had a fair degree of

1 minute left in chapter

luck on our side. Of course, a bettor should never judge the validity of their prediction model
by only one or a small handful of results, regardless of whether those results are favourable
or unfavourable. Good and back luck (those pesky hidden random variables) dominate in the
short term. However, what might be worth observing is generally how close the Monte Carlo
model’s implied odds were to the published bookmakers’ odds. Broadly speaking that should
give you confidence that you are probably doing the right things. Your task, as a modeller, is
then to fine-tune what you do to build something better than the models the bookmakers are
using. Here, that might mean thinking about how the two players had been performing in the
earlier rounds of this tournament, how tired they might be from the amount of time spent on
court, how they’d been playing the key points, or what the weather was like for the match-up
and hence the speed of the surface.

With regards to one of those factors, the way players play the key points, this can throw the
assumption of point independence into doubt. The validity of the Monte Carlo simulation rests
on every point played being independent of the previous ones, in the same way that spins ona
roulette wheel are. But arguably, the assumption of point independence is not secure because,
like football players, tennis players have memories fuelled by emotion, that will subject them
to all sorts of motivating factors depending on the point being played. Djokovic, in particular,
is renowned for being a master of playing the key points. One only has to think of the 2019
Wimbledon Final, where he saved two consecutive match points whilst returning the Roger
Federer serve, going on to claim the title. More generally, human beings are subject to many
cognitive biases; one of these is loss aversion, where we are more sensitive to losses than we
are to gains. It has been shown in professional golf, for example, that players are nearly 4%
more likely to save par than to make birdie, even after the influence of the shot distance
has been accounted for. Making birdie is considered a gain, understandably so because it is
one shot better than what the golfer should be achieving. Similarly, shooting a bogey will
psychologically be regarded as a loss. Thus, players try harder to avoid it. Perhaps a similar
loss aversion may be at play in tennis, with players saving a higher percentage of break points
than their service point win percentage would predict. Always be sensitive to possible flaws
in your model assumptions. Systematic mistakes in inputs are likely to be magnified once you
have arrived at the outputs, particularly in stepwise models like a Markov chain Monte Carlo
simulation, in the same way that a marginally better tennis player accumulates their advantage
over the course of a match.

A Monte Carlo Simulation for the NBA Finals
V’ll put my cards on the table right now: I know almost nothing about NBA basketball; its

rules, its history, its nuances. About the only things I know are that it’s a pretty high scoring
game (and seems to have got higher in the past 10 to 15 seasons), draws are not permitted,

22%




--- Page 14 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

with overtime settling games where necessary, and there’s a rather good Netflix series about
Michael Jordan’s Chicago Bulls (The Last Dance). Nevertheless, I wanted to have a go at using a
Monte Carlo simulation to model a game, or to be more precise, a series of games. I chose the
2018/19 Finals between the Toronto Raptors and defending champions Golden State Warriors.
The 2019/20 season was disrupted by the pandemic and the completion of games did not
follow the format of previous seasons, with only 22 of the 30 teams, who were within 6 games
of a playoffs spot when the season was suspended, invited to finish the season at the Disney
Bubble in Orlando. The NBA Finals are competed over a maximum of 7 games, with the first to
4 winning. Toronto held home court advantage, meaning they would play at home in games 1,
2,5 and 7, with Golden State playing at home in games 3, 4 and 6. If ateam wins 4 games before
the end of the 7-match series, the remaining games are not played. Game 1 was scheduled for
30 May 2019.

The model I used to calculate the relative attack and defence strengths of Toronto and
Golden State was the same as the one I used for the football ratings model when I attempted
to simulate the end of the 2019/20 Premier League season. Like football, the NBA has a home
court advantage, although it’s not as strong. Hence it makes sense to treat the home and away
metrics separately. Again, calculating these is simply a function of comparing the two teams’
average point scoring and conceding to the NBA average point scoring and conceding, and
finally calculating the expected points they will score. For this task I included the pre-season
games as well. For games where Toronto were scheduled to play at home (1, 2, 5 and 7) the
expected points for Toronto and Golden State were 111.34 and 111.61 respectively, with a game
total of 222.95. For games where Golden State were scheduled to play at home (3, 4 and 6), their
expected points were 112.49, and 112.69 for Toronto. According to my model, both home and
away games, and therefore the Series as a whole, were about as close to a coin toss as you could
reasonably get. This is not how the bookmakers saw it; more on that shortly.

How do we translate these expected points into win probabilities? In football, the
distribution of goals scored by a team can be approximated by the Poisson distribution. Points
scored in NBA matches, by contrast, are more readily described by the normal (or Gaussian)
distribution. Recall that Excel has a function NORM.DIST that allows one to calculate the
probability of either fewer than or more than x number of points, provided you know the
arithmetic mean and standard deviation. I’ve already quoted the means; these are just the
expected points as calculated by Dixon-Coles. Excel has the STDEV function to allow you
to calculate the standard deviation for a set of data. The standard deviation for Toronto’s
points when playing home in the 2018/19 season was 11.15. For Golden State playing away
it was 12.31. For example, the probability that Toronto will score fewer than 100 points at
home can be calculated in Excel by NORM.DIST(100,111.34,11.15,TRUE) = 15.45%. This is
equivalent to the area under the normal curve to the left of the position marked by 100
points. Similarly, the probability that Golden State will score more than 115 points away is
given by 1 —- NORM.DIST(115,111.61,12.31,TRUE) = 39.15%. This is equivalent to the area

1 minute left in chapter

under the normal curve to the right of the position marked by 115 points. Excel also has an
inverse function with the syntax NORM.INV(p,u,0) which, for a given probability, p, defined
by NORM.DIST(x,u,0, TRUE), returns the value x (where p is the mean and o the standard
deviation). Thus, if we randomise p using Excel’s RAND() function, we can randomise the
output x, where those outputs will be distributed normally. If we repeat this over a large
number of iterations for both teams, we’ve made our Monte Carlo simulation. It’s then just a
simple matter of counting the number of times Toronto beat Golden State (or vice versa) to
calculate the implied probability (and fair odds) of either team winning a game.

The frequency distribution of simulated points for both teams where Toronto plays at home
is shown in the chart below. I’ve used a scatter plot rather than a histogram because, with
the latter, you would be unable to tell them apart. They look just like the normal distribution
curve; they should, because the outputs were generated by a random variable with a normal
distribution. The model suggested the teams were very closely matched. Toronto’s distribution
is slightly narrower and taller than Golden State’s because their points standard deviation was
a little smaller. In 100,000 iterations, Toronto outscored Golden State 49,444 times, implying
their win probability was 49.444%, equivalent to fair odds of 2.025. Golden State’s win
probability was 50.556%, pricing them at 1.978.

Distribution of simulated points totals for Toronto and Golden
State where Toronto plays at home

4.0%

3.5%

3.0%

25% Toronto
a Golden State
2 2.0%
9g
> 1.5%
®
am 1.0%

0.5%

0.0%

60 70 80 90 100 110 120 130 140 150° 160

Points

In fact, these odds are pretty close to the bookmakers’ fair opening odds for Game 1. With the
margin removed, Pinnacle opened with 2.08 for Toronto and 1.92 for Golden State; bet365,
similarly, offered 2.05 and 1.95, respectively. However, in the build up to the match, the odds
for Toronto shortened quite significantly, such that by market closure and the start of the game
they were favourites, with the fair odds 1.83 and 1.80 (and Golden State 2.21 and 2.25) at
Pinnacle and bet365, respectively. I have no idea what new information prompted such a move.

22%




--- Page 15 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

Whatever it was, it evidently proved to be reliable. Toronto won the game and indeed went on
to win the whole Series 4-2.

I also ran my Monte Carlo simulation for the remaining games, up to a possible maximum
of 7 in total. When Toronto played away, they had expected points of 112.69 (with a standard
deviation of 11.35). Golden State playing at home had 112.49 points (with a standard deviation
of 12.62). Toronto won 50.421% of these simulated games, implying odds of 1.983. Golden
State had implied odds of 2.017. For the Series as a whole, Toronto won 49.922% of the
simulated NBA Finals, implying their odds for victory were 2.003. Golden State would have
been priced at 1.997.

In stark contrast to my model, pundits and bookmakers had made Golden State firm
favourites to defend the title they had won the season before. Several days prior to Game 1 after
the finalists became known, the market opened at 1.36 (fair price 1.40) and 3.25 (fair price 3.5)
for Golden State and Toronto, respectively. I suspect, in line with the odds movement for Game
1, these narrowed somewhat, but I don’t have the data to check. Nevertheless, with respect to
the overall Series winner, my model was evidently missing something. A big clue comes from
the implied prices of games 3, 4 and 6, where Golden State were at home. My model actually
made Toronto marginally favourite in those games, whilst the bookmakers believed Golden
State, with an opening fair price of 1.50, had twice the chance of Toronto (3.00) to win Game
3. Whilst that gap narrowed by market closing (Golden State 1.74, Toronto 2.36), this was
still a long way from my model outputs. Toronto won Game 3, yet the bookmakers (and fans)
still believed Golden State would start cranking up their performance as the Series progressed,
pricing them at 1.49 for Game 4. In the event, that didn’t happen, with Toronto winning this
game as well, before finally claiming the Series in Game 6 away from home.

Why did my model diverge so significantly from the ones the bookmakers were using for
Golden State playing at home? I used the whole of the preseason and regular season to calculate
the attack and defence strengths, and the expected points for each team. Perhaps this time
frame is too long. Much can change over the course of the season. As for my Premier League
model, I gave no preferential weighting to more recent games. Another clue that my model was
inaccurate came from the actual number of points it was projecting the two teams to score.
With a total of nearly 223 expected for Toronto’s home games, that was a whole 10 points
higher than the middling line - 213 points — on the over/under betting market for Game 1,
where a bet on either over or under 213 points has roughly a 50% chance of success. Based
on my Monte Carlo model, the implied probability of seeing under 213 points was 27.6%, and
72.4% for over. Again, why such a large discrepancy?

There is a well-worn maxim that describes the postseason: ‘The Game Slows Down.’
Apparently, this is more than a mere cliché. Statistical analysis proves that the pace in the
NBA playoffs does slow slightly, by about 2%, although the precise reasons for this remain
somewhat of a mystery. I’m sure readers more familiar with the NBA will have their views
about what they might be. This does not mean the relative quality of offence or defence

1 minute left in chapter

changes significantly; better offences and defences, after all, make the playoffs. ‘Slowing down’
implies that the pace of play is slower, with possessions lasting longer, teams using more of the
permitted 24-second shot clock per possession, and hence fewer possessions per 48 minutes of
play (which is the length of a game). Looking at the game totals for the 2018/19 season, the
average postseason game total prior to the Finals was 213.30 points, compared to a figure of
220.84 for the preseason and regular season combined. I decided to rerun the model, this time
using only the playoffs points to calculate the attack and defence strengths, and the expected
points, for both teams. Their sum was now considerably lower and closer to the bookmakers’
total points markets. For Toronto home games, Toronto had 105.27 expected points whilst
Golden State had 105.14. Again, close to a coin toss, but the bookmakers’ odds were implying
that anyway. For Golden State home games, by contrast, Toronto had 106.71 but Golden State
had more than 4 more, with 110.89, making them favourites. By how much?

Let’s run the Monte Carlo simulation again, but first, a little aside. In theory I should use
the standard deviation for Toronto and Golden State’s postseason game points but given the
small number of games, this led to some rather unreliable figures. For example, Toronto’s
away points spread in the postseason was very narrow (with away scores of 98, 107, 95, 101,
101, 100, 103 and 105) having a standard deviation of just 3.81. This is almost certainly a
lucky occurrence and would not be expected to continue over a much longer period of games
played under ‘playoff conditions’. Using this figure to randomly generate their match totals for
the simulation via the NORM.INV function, however, would mean a much narrower range of
outputs than would typically be expected for an NBA team. Thus, to make the simulation more
reliable, I decided to stick with the same standard deviation figures I had used for both teams
in the original model, that is those for the whole season. We could argue all day about exactly
what figures should be used. For example, slightly smaller expected totals should probably be
accompanied by a slightly smaller spread in those scores. Nevertheless, the discrepancy will be
marginal and make little difference to the final Monte Carlo outputs. Whatever they should be,
it makes sense for them to be around 11 to 12, since that is the average standard deviation for
the NBA more generally, even in the playoffs.

So, to the results. Toronto were winning 50.443% of home games (implied odds 1.982),
Golden State winning 49.557% away (implied odds 2.018). The big change, however, was
when Golden State played at home. Now they were winning 59.705% (implied odds 1.675),
with Toronto away winning 40.295% (implied odds 2.482). These odds are much closer to the
quoted closing odds for Game 3 as I reported them earlier, although something like a 15%
expected value was still available in the opening market before those shortened. You can see the
difference between the two teams more markedly in the points distribution below.




--- Page 16 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

Distribution of simulated points totals for Toronto and Golden
State where Toronto plays away ('Playoffs-only' model)

Toronto
Golden State

Frequency

60 70 80 90 100 110 120 130 140 150 160
Points

Since points scored by NBA teams are normally distributed, there is actually a little trick
one can use to calculate the probability of one team outscoring another, and thus winning
a game, avoiding the need for a Monte Carlo simulation. Provided the scoring of points by
Toronto (Xtoronto) and Golden State (xcojgenstate) are independent, then the points differences
between teams (Xgitference) are also normally distributed, with a mean (Ugigference) given by the
difference between the teams’ means, and a standard deviation (Ogifference) given by the square
root of the sum of the squares of the individual team standard deviations. Mathematically, we
would write:

Xdifference — XToronto — *GoldenState

difference = HToronto — HGoldenState

— / 2 2
Odifference = V OToronto” + OGoldenState

The standardised form of this distribution is then:

_ “difference — Ldifference
Zdifference — Ouift
ifference

where Zgifference is the number of standard deviations xgigerence is away from the mean
Udifference: The probability of Toronto scoring fewer than Golden State (xqoronto — XcoldenState
< 0) can be calculated in Excel using the NORM.DIST function, setting xgitference to O and use
Udifference And Ogisference aS the other function arguments. Doing the arithmetic, we have:

1 minute left in chapter

Ldifference = 106.71 110.89 = -4.18

difference = ¥ 11.35? + 12.62? = 16.98

Thus, NORM.DIST(-4.18,16.98,TRUE) = 0.59724 (or 59.724%). Following the complementary
rule, the probability of Toronto scoring more than Golden State is then simply 1 - 0.59724 =
0.40276 or 40.276%. Recall that the figure from the Monte Carlo simulation was 40.295%. I’ve
drawn the frequency distribution of the 100,000 simulated points differences between Toronto
(away) and Golden State (home). The mean is -4.18 points, the standard deviation is 16.98, and
the area under the ‘curve’ to the right of 0 points difference is 0.40276, or 40.276% of the total
area under the whole ‘curve’. That area is equivalent to the probability of Toronto scoring more
than Golden State when playing away.

Distribution of points differnces beween Toronto and Golden
State where Golden State plays away ('Playoffs-only' model)

3.0%

y 50
2.5%

Frequency

0.0%
-90 -80 -70 -60 -50 -40 -30 -20-10 0 10 20 30 40 50 60 70 80 90

Points difference

I’'d understand completely if the mathematics and terminology in this last paragraph seem
a little inscrutable. When it proves to be so, that, after all, is why we use the Monte Carlo
simulation.

With my updated ‘playoffs-only’ model, Golden State were now winning 58.783% of the 7-
match Series (implied Series winner odds 1.701) with Toronto 41.217% (implied odds 2.426).
These are still narrower than the published odds several days before Game 1, but, as mentioned,
I suspect those published odds narrowed too, in line with the shortening of Toronto for Game 1.
Nevertheless, a market opening price of 3.50 offered expected value of 44.3% (3.50/2.426 — 1)
with the playoffs-data-only model and an even more generous 74.7% (3.50/2.003 - 1) with the
full-season-data model. Figures like those would normally have me worrying that my model
was wrong. Just possibly, however, it was the early market opinion that was in error this time,
underestimating Toronto’s true chances of becoming Champions, blinded by Golden State’s

24%




--- Page 17 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

dynastic run of five consecutive Finals (winning 3 of them between 2015 and 2018). Evidently,
the price swings towards Toronto in the lead up to all but one of the games (Game 5, which
Golden State had to win to continue the Series) would support the argument that Toronto were
increasingly being regarded as serious contenders. They did, after all, win all three of their away
games, where Golden State were supposed to be heavy favourites. Of course, I don’t really know
anything about the subtleties of NBA, so perhaps it’s best to leave speculation about all of this
to those who do.

A Monte Carlo Simulation for the US Presidential Election

It is almost certain that the 2020 US presidential election between Republican incumbent
Donald Trump and Democrat challenger Joe Biden became the biggest betting market of all
time. Matthew Shaddick, the head of political betting at Ladbrokes Coral Group, had estimated
about £1bn would be wagered globally across the industry. I suspect that underestimates the
real figure by at least an order of magnitude. The Betfair exchange alone saw approaching £2bn
of matched bets. Given the level of excitement that was evidently being caused by arguably the
most divisive President the United States has seen, on the night of the election I felt motivated
to throw my hat into the prediction modelling ring. At this point I should declare that I had
believed Joe Biden a stonewall favourite to win the election. When I backed him, the price
was 1.51 on the Betfair exchange. I considered the ‘true’ price to be nearer 1.20, meaning
that if I was correct, I was being offered around 25% expected value. Why? Whilst the polls
had called the wrong result in 2016, there were a number of factors that suggested history
would not repeat itself. Firstly, Biden had a bigger poll lead than Hillary Clinton, particularly
in the swing States. Secondly, Biden is not Hillary Clinton; she was evidently not loved by a
significant section of the American electorate. Thirdly, Biden is not Trump; for many this would
prove to be an anti-Trump vote rather than pro-Biden one. Fourthly, contrary to the Trump
narrative, Biden is neither a progressive nor a socialist, but rather a moderate in the Democratic
Party. Fifthly, COVID-19 and the resulting economic recession may have sowed seeds of doubt
amongst swing voters about Trump’s competence in handling a crisis and perhaps more
importantly, the direction of his moral compass implied by his failings in this regard. Finally, as
culturally regressive as it sounds, perhaps it just needed a 70-something white man to defeat a
70-something white man.

AUS presidential election is decided by the Electoral College, a group of presidential electors
required by the US Constitution to form every four years for the sole purpose of electing the
President. Each state appoints a number of electors which broadly reflect the size of the state by
population. California, for example, is the largest with 55; the smallest like Vermont, Delaware
and the District of Columbia have only 3. It is the Electoral College, rather than the public
electorate, who choose the next President. Currently, a total of 538 electors are appointed,

1 minute left in chapter

meaning an absolute majority of 270 or more Electoral College votes is required to elect the
President. With a couple of exceptions, Maine and Nebraska, which split their votes rather than
awarding all to the winner, convention has it that a state’s Electoral College electors all vote
according to the winner of the public vote in that state.

The prediction website FiveThirtyEight (named after the number of Electoral College votes
up for grabs in a US Election) had been even more confident about a Biden win, predicting
an 89% chance (with implied odds of 1.124) when they stopped updating their model on
the day of the election, with Biden taking an expected 348 of the Electoral College votes (I
will abbreviate hereafter with ECVs) and Trump the remaining 190. The Economist’s election
model was even more lopsided towards Biden, estimating a margin of victory by 356 to 182.
The New Statesman model predicted 339 votes to 199 in favour of Biden. By contrast, the
spread betting firm Sporting Index were predicting a Biden win with 308 EVCs. Who was right;
I wanted to find out, and more importantly, attempt to correlate projected ECVs with projected
winner odds. Such a job is perfectly suited to a Monte Carlo simulation.

To build a Monte Carlo simulation, I first needed an input model to predict the expected
probability of either Biden or Trump winning the Election. I could build my own forecasting
model. Unfortunately, I know even less about US politics than I do about the NBA. Furthermore,
after 25 years of trying to build them for a football betting market, I’ve come to appreciate that
it’s actually easier to copy what the experts - the bookmakers - do. However, if we are trying
to find out if the bookmakers, rather than the pollsters (and I), are wrong, we can’t exactly
use their betting odds, can we? We need a proxy source of data and a proxy forecast model.
One method is to use the state betting market, reciprocating the betting odds to calculate the
probabilities for each of the 50 States plus the District of Columbia, and sum the expected
Electoral College votes (xECV) which they imply. For example, Trump was considered to have a
61% chance of taking Florida. With 29 Electoral College votes up for grabs that is equivalent to
17.7 xECV (29 x 0.61), with 11.3 going to Biden (29 x 0.39). The Betfair exchange had just such
a market for all of the 50 (+1) states. I collected their odds at 23:45 on Tuesday 3 November
2020, 15 minutes before voting closed on the eastern seaboard, removed any margin that
was present, and did the maths. Biden’s total xECV was 307.75, with Trump gaining 230.25,
essentially matching the Sporting Index spread betting market. The full set of probabilities and
calculated xECVs for each state is shown in the table below, ranked in ascending order of a
Trump win probability.




--- Page 18 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

How do we translate these xECVs into the odds for Biden or Trump to win? Enter the
Monte Carlo simulation. By using a uniformly distributed random number, in Excel the RAND()
function, to simulate the result of each state, we can run the election many times and count
how often Biden or Trump win. If the random number falls below a win probability for a
candidate, they take all the ECVs; if above, they take none. For the purposes of this model, I
assumed Maine and Nebraska behaved as the other 49. They only contribute 4 and 5 votes
respectively to the total of 538, so won’t make much meaningful difference to the model
outcome. The first chart below shows the distribution of Biden/Trump ECV tallies from
100,000 simulated elections. Biden won 89.9% of the simulated elections, which implies true
odds of 1.125, almost the same as FiveThirtyEight’s expected win probability for Biden. The
average ECV tally for Trump was 230.29, for Biden, 307.71, almost identical to the calculated
xECVs above, as they should be. The standard deviation in ECVs was 31.95 ECVs for Biden and
Trump alike (clearly so since the sum of ECVs for the two candidates on any model iteration
always makes 538).

Massachusetts | 1.71% | 9829% 1/11. | 019
Califomia. |: 189% |9811% 155.
New York
Rhode Island

Monte Carlo distribution of Electoral College votes
1.6%
1.4%

1.2%

Republicans Democrats

Biden
ECV

Frequency

0.4%
0.2%
0.0%

| Arizona

iz
| North Carolina
i

0) 50. 100 150 200 250 300 350 400 450 500

Electoral College votes

The frequency distribution of both Biden and Trump ECVs closely follows the normal
distribution. That is to be expected, given their randomised construction. However, there is
evidently a problem with my model. FiveThirtyEight predicted 348 Electoral College votes for
Biden and 190 for Trump as the best estimate in their 40,000-iteration Monte Carlo simulation.
Mine was predicting 308 versus 230, yet with the same odds for a Biden victory. The
discrepancy arises because my model considered the public vote, and by extension the ECVs,
in one state to be independent of all other states. Unsurprisingly, that is a flawed assumption.
| Clearly there are common nation-level correlations at play, meaning that if one state, for
oo7 | example Wisconsin, votes in a particular way, another, for example Michigan, is likely to do so

| West Virginia

| Oklahoma
iS
LT

1 minute left in chapter 25%




--- Page 19 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

too. A prediction error in one state is thus also likely to be correlated in others.

The name for this directional relationship between two random variables is covariance.
Covariance is the reason bookmakers will not allow you to build multiple bets (doubles, trebles
and so on) from state betting markets, or in any other betting market for that matter, where
the outcome probabilities are correlated. The most significant impact of a covariance between
random variables - in this case win probabilities in individual states - is a much greater
spread or variance in the data. The greater the variance, the less likely one candidate will
beat the other. This is simply a consequence of the laws of probability. You can visualise this
in the chart above. The darker region of overlap between the two distributions is equivalent
to the probability of Trump beating Biden. If their distributions were taller and narrower,
with a smaller standard deviation but the same means, this area of overlap would shrink. In
contrast, if the distributions were shorter and wider with a larger standard deviation, the area
of overlap would grow. FiveThirtyEight presumably make it their business to develop a highly
sophisticated covariance model that considers the state-by-state voting correlations. Their ECV
distribution had nearly double my standard deviation, at almost 60.

Undeterred, I made another attempt at building a US election prediction model, this time
attempting to introduce some state covariance, in much the same way that I described earlier
how fair odds might be transformed into unfair ones using a common exponent. My method
was thus as follows. For every iteration in the simulation, the Betfair-implied win probabilities
for Trump in each state were raised to the power of a specific exponent. For example, if the
exponent was 1.1 then Trump’s win probability in New York was transformed from 0.0248 to
0.02481.1 = 0.0171; for Arizona, from 0.4527 to 0.45271-1 = 0.4182; for Alabama, from 0.9898
to 0.98981-1 = 0.9888. Biden’s transformed win percentages were then calculated using the
complementary rule (1 —- Trump’s win percentage). In this way, the error in the Betfair model
was quantified by the size of the transformation exponent and correlated across all states. An
obvious weakness is that it’s likely not all states will exhibit the same level of covariance; some
will show more independence, others less, but I have neither the intellect nor the financial
resources that FiveThirtyEight have at their disposal to account for these differences. Across
the 100,000 Monte Carlo iterations, the size of the transforming exponent was then
randomised using a lognormal distribution (where the logarithm of a random variable is
normally distributed - I'll introduce this distribution more formally in the chapter on staking),
with a mean of 1. Anything higher than 1 would collectively reduce Trump’s win probabilities
in all states, as in the example above. Anything lower than 1 (but always above 0) would
collectively increase Trump’s win probabilities. The point of using a lognormal rather than
normal distribution is to ensure symmetry between exponents above and below 1. Excel has
the function LOGNORM.INV(p,y,o) to perform this task. By randomising the probability
argument, p, with RANDQ, I could thus lognormally randomise the exponents. Then, by
varying the size of the standard deviation, o, of this distribution (I chose values between 0 and
1) I could vary the strength of the covariance. The bigger the standard deviation, the more

1 minute left in chapter

varied the exponents, and the greater the impact of correlated errors in the implied Betfair
probabilities. Yes, I know this all sounds rather convoluted, I’m sorry, and I rather suspect it
would fail any statistical modelling course. Anyway, the model outputs are arguably meant to
be more qualitatively informative than quantitatively valid.

The table below compares 11 different simulations, for 11 different values of my lognormal
standard deviation. A figure of 0 means all original Betfair state win probabilities remain
untransformed, since every exponent is 1. This is effectively equivalent to the original model,
where state voting is completely independent. You can see that as the level of covariance
is increased (by increasing the lognormal standard deviation), the amount of spread in the
distribution of ECVs (as measured by the ECV standard deviation) increases. This is entirely
predictable. Since state covariance means that states tend to vote the same way, this will lead
to bigger ECV tallies for Trump or Biden, and bigger margins of victory in more of the Monte
Carlo iterations. This increase in distribution spread is matched by a decrease in the chances of
Biden taking overall victory, for the reasons explained earlier. Hence, his implied odds lengthen
too. You will also notice that with a lengthening of price, Biden’s mean ECV total (equivalent to
xECV) starts to decrease, although the median ECV broadly remains unchanged. My covariance
model has introduced an asymmetry into the distribution of ECVs, which increases as more
covariance is applied. Trump has proportionally more elections than Biden where he scores
a big win. I think this arises because, in his luckiest elections, there are more available ECVs
available for Trump to overturn in Democratic strongholds like California and New York than
vice versa. The stronger the covariance, the more likely it becomes that Trump can overturn
these states in blocks, and the bigger his ECV tally. Some big wins will skew the arithmetic
mean above the median.




--- Page 20 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

Trump Biden

| Lognormal |

i mean | i i | median |

ECV ECV

So which covariance is the right one? Clearly, there’s a certain amount of subjectivity in
answering that, but we do at least have one independent measure: the ECV standard deviation
in FiveThirtyEight’s simulation. Recall, this was nearly 60. Looking at the table above, the
closest match to that figure is in line 7. For this simulation, Biden’s mean ECV tally was 302,
with the median (or middle) tally a little higher at 306. Trump’s mean and median were 236
and 232. From this simulation run, Biden had a win probability of 72% with implied odds of
1.388. The frequency distribution is shown below. I’ve purposely maintained the same vertical
axis scale so you can compare this to the original distribution, where state voting was wrongly
assumed to be completely independent. Covariance increases the standard deviation, thus
making the distributions fatter and shorter, although their total areas remain the same as for
the independent model, adding up to 1 (or 100%). You can also see the positive skew in Trump’s
distribution, with a longer, fatter tail towards the right hand side, increasing the mean relative
to the median, and vice versa for Biden’s. The central darkest shaded area is now considerably
larger, and Trump has a greater probability of winning.

1 minute left in chapter

Monte Carlo distribution of Electoral College votes
(covariance model)

6%
4%

29 0

Republicans Democrats

0%
0.8%
0.6%
0.4%
0.2%
0.0%

Frequency

0) 50. 100 150 200 250 300 350 400 450 500

Electoral College votes

We are now perhaps in a better position to attempt an answer to the original question: who was
right about the US Election - the pollsters, who were predicting about 350 ECVs for Biden witha
90% chance of winning, or the betting models, who were projecting about 306 to 308 ECVs and
around a 70% chance for victory? As I’ve said before, it’s never a good idea to judge the quality of
a prediction by a single outcome, since we never get to see the real election played out 100,000
times, like we can on a computer. The single real-world result that actually happens could well
just be lucky. We will ultimately never know. Nevertheless, it’s always rewarding when your
model predicts the outcome accurately, and that is what has happened here. Biden won the
US Election with 306 ECVs to Trump’s 232, matching exactly the median in the covariance
distribution above, and just 4 away from the arithmetic mean. Similarly, the figure of 308 for
Biden’s xECV, using just Betfair’s state betting odds and no Monte Carlo simulation, is also close,
far closer than FiveThirtyEight and the other pollsters. FiveThirtyEight would remind us that
their projected figure of 348 is within one standard deviation of Biden’s actual result, and thus
well within typical error margins associated with significance testing, where a difference of at
least two standard deviations is typically required before one would begin to suspect the model
was invalid. Nevertheless, intuition tells me that the betting markets have done a better job at
forecasting the outcome than the pollsters.

In recent times, it has become almost a mantra to claim that the betting markets offer the
best political polling predictions. Whilst their wisdom can also be flawed from time to time, as
was the case in the 2016 presidential election and for Brexit, there seems to be some validity to
the claim. It is based on the observation that betting markets, more than any other prediction
methods, are informed not only by considerable professional insight, as for the traditional
polls, but also by weight of money. Money is a means of quantifying opinions. It’s also required

26%




--- Page 21 ---

MONTE CARLO OR BUST: SIMPLE SIMULATIONS FOR ASPIRING SPORTS BETTORS

to make a livelihood. If your livelihood is earned by expressing opinions about the future, but
where you only get paid if you're right, there is a certain pressure not to be wrong. This is called
having skin in the game. Arguably, whilst pollsters may lose reputation and face for making
bad calls, they will, generally speaking, still keep their salaries. Bettors, and in particular,
professional bettors, however, depend entirely on the quality of their predictions. If they are
wrong, they don’t get paid, and their families don’t get fed.

There is one more outstanding question that I need to answer: if the Betfair state betting
market, via my Monte Carlo simulation, was implying Biden odds of 1.39, why was their main
winner market offering me a price of 1.51, and an expected value of 8.6%? There appears to
be some disconnect between the two markets. At least one of them must have been inefficient,
that is to say, inaccurate. Granted, it seems I was wrong about Biden’s true price being closer to
1.20, but if my Monte Carlo model was correct, 1.51 was still a positive expectation bet. Indeed,
at the time I collected Betfair’s state betting odds they were quoting 1.68 for Biden to win. Why
would Betfair’s winner market be at odds (pardon the pun) with their state betting market?
Given the much smaller size of the latter in terms of the volume of money wagered, it’s all the
more puzzling that the former seems to have been less accurate. More usually, market volume
correlates with market efficiency. More money means more opinions; more opinions mean
more cancelling of random errors; more error cancellation means more ‘wisdom of a crowd’;
more ‘wisdom of a crowd’ means a more accurate betting price. Of course, I can’t entirely
rule out the possibility that either the way I transformed Betfair’s state betting market into

1 minute left in chapter

a presidential winner probability, or the state betting probabilities themselves, were wrong.
Interestingly, however, if we just look at the odds for the 51 betting markets, only one, Georgia,
saw the favourite (Trump) lose. In all the other closely fought States —- North Carolina, Florida,
Ohio, Arizona, Wisconsin, Minnesota, Pennsylvania, Texas and Michigan - the Betfair state
betting market called the most likely outcome correctly.

Why wasn't Donald Trump allowed back into the White House?
Because it's forBiden. Thank you and goodnight.

P.S. That is a joke, not a conspiracy.

I think that’s enough prediction modelling. I’ve tried to keep things as simple as possible, but
I should apologise (again) if I’ve lost some of you along the way, or conversely if you’ve found
much of this chapter too rudimentary. For those in the latter camp, I can strongly recommend
Andrew Mack’s Statistical Sports Models in Excel, a two-part volume that goes into much more
mathematical and Excel programming detail than I wanted to do here. Andrew looks at all
kinds of traditional modelling approaches, not just Monte Carlo. Anyway, time to move on.
Next, we’ll look at how the Monte Carlo simulation can help you understand your likelihood of
winning and making a profit.


