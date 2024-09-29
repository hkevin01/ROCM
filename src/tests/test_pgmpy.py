from pgmpy.models import BayesianNetwork  # Import the BayesianNetwork class for creating the model
from pgmpy.inference import VariableElimination  # Import the VariableElimination method for inference
from pgmpy.factors.discrete import TabularCPD  # Import TabularCPD for defining conditional probability distributions

# Define the Bayesian Network structure
model = BayesianNetwork([('ClosedShape', 'DigitZero'), 
                          ('StraightLines', 'DigitZero')])
# This creates a Bayesian network with directed edges indicating that 
# 'ClosedShape' and 'StraightLines' influence 'DigitZero'.

# Define the Conditional Probability Distributions (CPDs)

# P(ClosedShape)
cpd_closed_shape = TabularCPD(variable='ClosedShape', 
                               variable_card=2, 
                               values=[[0.3], [0.7]])  
# This defines the probability distribution of 'ClosedShape':
# P(ClosedShape=False) = 0.3, P(ClosedShape=True) = 0.7

# P(StraightLines)
cpd_straight_lines = TabularCPD(variable='StraightLines', 
                                 variable_card=2, 
                                 values=[[0.4], [0.6]])  
# This defines the probability distribution of 'StraightLines':
# P(StraightLines=False) = 0.4, P(StraightLines=True) = 0.6

# P(DigitZero | ClosedShape, StraightLines)
cpd_digit_zero = TabularCPD(variable='DigitZero', 
                             variable_card=2, 
                             values=[[0.9, 0.8, 0.6, 0.1],  # P(DigitZero=False | ClosedShape, StraightLines)
                                     [0.1, 0.2, 0.4, 0.9]], # P(DigitZero=True | ClosedShape, StraightLines)
                             evidence=['ClosedShape', 'StraightLines'],  # The evidence nodes
                             evidence_card=[2, 2])  # Both evidence nodes are binary

# Add CPDs to the model
model.add_cpds(cpd_closed_shape, cpd_straight_lines, cpd_digit_zero)
# This adds the CPDs to the Bayesian network model.

# Check if the model is valid
assert model.check_model()  # Validates the model structure and CPDs

# Perform inference using Variable Elimination
inference = VariableElimination(model)

# Example query: What is the probability that the digit is '0' 
# given that it is a closed shape and has straight lines?
result = inference.query(variables=['DigitZero'], 
                         evidence={'ClosedShape': 1, 'StraightLines': 1})
# Here, '1' indicates True for both 'ClosedShape' and 'StraightLines'.

# Print the result of the query
print(result)  # Outputs the probability distribution of 'DigitZero'