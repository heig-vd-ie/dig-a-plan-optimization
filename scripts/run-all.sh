#!/bin/sh

# Set this variable to true to run only the first four experiments
FIRST_ONLY=${FIRST_ONLY:-false}
DELAY_TIME=10

echo "Start Run"

# First experiment
python experiments/expansion_planning_script.py --kace ieee_33 --fixed_switches true --cachename ieee33_fixed_switches
echo "Finished ieee33 fixed switches"
sleep $DELAY_TIME
# Second experiment
python experiments/expansion_planning_script.py --kace ieee_33 --cachename ieee33_expectation
echo "Finished ieee33"
sleep $DELAY_TIME
# Third experiment
python experiments/expansion_planning_script.py --kace ieee_33 --riskmeasuretype Wasserstein --cachename ieee33_wasserstein
echo "Finished ieee33 wasserstein"
sleep $DELAY_TIME
# Fourth experiment
python experiments/expansion_planning_script.py --kace ieee_33 --riskmeasuretype WorstCase --cachename ieee33_worstcase
echo "Finished ieee33 worstcase"
sleep $DELAY_TIME
# If FIRST_ONLY is true, exit here
if [ "$FIRST_ONLY" = true ]; then
    echo "stopping after first four experiments."
    exit 0
fi

# Additional experiments
python experiments/expansion_planning_script.py --kace boisy --feedername feeder_1 --fixed_switches true --cachename boisy_feeder_1
echo "Finished feeder_1"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace boisy --feedername feeder_2 --fixed_switches true --cachename boisy_feeder_2
echo "Finished feeder_2"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername aumont --fixed_switches true --cachename estavayer_aumont
echo "Finished aumont"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername autoroutes --fixed_switches true --cachename estavayer_autoroutes
echo "Finished autoroutes"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername bel-air --fixed_switches true --cachename estavayer_bel-air
echo "Finished bel-air"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername centre_ville --fixed_switches true --cachename estavayer_centre_ville
echo "Finished centre_ville"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername st-aubin --fixed_switches true --cachename estavayer_st-aubin
echo "Finished st-aubin"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername tout_vent --fixed_switches true --cachename estavayer_tout_vent
echo "Finished tout_vent"
sleep $DELAY_TIME
python experiments/expansion_planning_script.py --kace estavayer --feedername zone_industrielle --fixed_switches true --cachename estavayer_zone_industrielle
echo "Finished zone_industrielle"
