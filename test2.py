"""
About this simulation:
This is a simulation of manufacturing process of semi conductors and we are going to use it
for the purpose of scheduling. Our goal is to train a RL based agent that can produce most optimized schedules
to maximize throughput and minimize tardiness.

In this simulation we have resources "machines"
machines are availble as instance.useable_machines only if
1- machine is free at time t
2- at time t there are lots that need that machine

if the above conditions are met, the machine is available for scheduling.

the agent will have choice to either deploy the lot 'work' on the machine or hold it for a while for greater good
in this simulation lots are collections of steps and at time we only use the lot.actual_step to determine the machine
our plan is to provide the agent with the following information:

The the lot must be processed along with all the lot.remaining_steps and its features to extract embeddings
this is done to have a rich information tensor for the lot instead od conventional method of proving only the actual_step/step_t

we will also try to develope the machine embeddings by looking into the future requirements of that specific machine. a rich tensor will be
tried to develop. once that part is done. we will perform some sort of connection between both tensor so pass that to the 
agent. so we have a combined embeddings and the actor will base on that information.
"""