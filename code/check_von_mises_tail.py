"""Conservative log-scale tail bound for the manuscript's von Mises scenarios."""
import math

# I_l(k)/I_0(k) <= (k/2)^l/l! * exp(k^2/[4(l+1)]), since I_0(k)>=1.
# For k<=20 and l>=257, use a geometric bound with squared ratio (10/258)^2.
l=257
log10_first=l*math.log10(10)-math.lgamma(l+1)/math.log(10)+100/(l+1)/math.log(10)
log10_tail=2*log10_first-math.log10(math.pi)-math.log10(1-(10/258)**2)
assert log10_tail < -500
print('Upper-bound log10 of omitted squared L2 tail:',log10_tail)
print('The analytic estimate is evaluated in floating point, not interval arithmetic.')
