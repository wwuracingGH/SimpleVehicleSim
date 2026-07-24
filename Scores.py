import numpy
from typing import Callable

class Event:
    def __init__(this, scores : list[float], scorefunc : Callable[[float], float]):
        this.scores_origional = scores
        this.scores_sorted = sorted(scores, reverse=True)
        this.scorefunc = scorefunc
    
    def get_placement(this, time : float):
        score = max(this.scorefunc(time),0)
        placement = next((i for i,d in enumerate(this.scores_sorted) if d < score)) + 1
        return placement

    def get_score(this, time : float):
        return this.scorefunc(time)


def print_event_results(eventname : str, time : float, years : list[CompetitionScores]):
    print('You completed the ' + eventname.split(' ')[0] + ' event in ' + str(round(time * 100) / 100) + ' seconds!')
    print('Year\tScore\tPlacement')
    for sc in years:
        print(f'{sc.year}\t{sc.EventScore(eventname, time):.2f}\t{sc.EventPlacement(eventname, time):d}')
    print()

def print_all_events(cost_s, presentation_s, design_s, accel_t, skidpad_t, autox_t, enduro_t, enduro_kwh,  years : list[CompetitionScores]):
    for sc in years:
        cost_p = sc.EventPlacement(CompetitionScores.NAMES_COST, cost_s)
        design_p = sc.EventPlacement(CompetitionScores.NAMES_DESIGN, design_s)
        presentation_p = sc.EventPlacement(CompetitionScores.NAMES_PRES, presentation_s)

        print(f'In {sc.year}, you got {cost_p}th in cost, {design_p}th in design and {presentation_p}th in presentation')

        accel_p   = sc.EventPlacement(CompetitionScores.NAMES_ACCEL, accel_t)
        skidpad_p = sc.EventPlacement(CompetitionScores.NAMES_SKID, skidpad_t)
        autox_p   = sc.EventPlacement(CompetitionScores.NAMES_AUTOX, autox_t)
        enduro_p  = sc.EventPlacement(CompetitionScores.NAMES_ENDURO, enduro_t)
        eff_p     = sc.EventPlacement(CompetitionScores.NAMES_EFF, sc.EfficiencyFactor(enduro_t, enduro_kwh))

        accel_s = sc.EventScore(CompetitionScores.NAMES_ACCEL, accel_t)
        skidpad_s = sc.EventScore(CompetitionScores.NAMES_SKID, skidpad_t)
        autox_s = sc.EventScore(CompetitionScores.NAMES_AUTOX, autox_t)
        enduro_s = sc.EventScore(CompetitionScores.NAMES_ENDURO, enduro_t)
        eff_s = sc.EventScore(CompetitionScores.NAMES_EFF, sc.EfficiencyFactor(enduro_t, enduro_kwh))

        total = accel_s + skidpad_s + autox_s + enduro_s + eff_s + cost_s + design_s + presentation_s

        print('event\t\tscore\tplacement')
        print(f'acceleration\t{accel_s:.2f}\t{accel_p}')
        print(f'skidpad\t\t{skidpad_s:.2f}\t{skidpad_p}')
        print(f'autocross\t{autox_s:.2f}\t{autox_p}')
        print(f'endurance\t{enduro_s:.2f}\t{enduro_p}')
        print(f'efficiency\t{eff_s:.2f}\t{eff_p}')

        print()
        print(f'Overall, your score was {total:.2f} and your placement was {sc.EventPlacement(CompetitionScores.NAMES_TOTAL, total)}')
        print()

class CompetitionScores:
    NAMES_COST   = 'Cost Score'
    NAMES_PRES   = 'Presentation Score'
    NAMES_DESIGN = 'Design Score'
    NAMES_ACCEL  = 'Acceleration Score'
    NAMES_SKID   = 'Skid Pad Score'
    NAMES_AUTOX  = 'Autocross Score'
    NAMES_ENDURO = 'Endurance Score'
    NAMES_EFF    = 'Efficiency Score'
    NAMES_TOTAL  = 'Total Score'
    
    def __init__(this, filepath : str, year : str):
        this.year = year

        with open(filepath) as f:
            cols = {}
            cols_idx = {}
            col_names = f.readline().replace("\n", '').split(',')
            
            for i,c in enumerate(col_names):
                cols[c] = []
                cols_idx[c] = i

            min_times = f.readline().split(',')
            
            this.mintime_accel  = float(min_times[cols_idx[this.NAMES_ACCEL]])
            this.maxtime_accel  = this.mintime_accel * 1.5
            
            this.mintime_skid   = float(min_times[cols_idx[this.NAMES_SKID]])
            this.maxtime_skid   = this.mintime_skid * 1.25
            
            this.mintime_autox  = float(min_times[cols_idx[this.NAMES_AUTOX]])
            this.maxtime_autox  = this.mintime_autox * 1.45
            
            this.mintime_enduro = float(min_times[cols_idx[this.NAMES_ENDURO]])
            this.maxtime_enduro = this.mintime_enduro * 1.45

            this.mintime_eff    = float(min_times[cols_idx[this.NAMES_EFF]].split('|')[0])
            this.minco2_eff     = float(min_times[cols_idx[this.NAMES_EFF]].split('|')[1])

            this.minfac_eff     = float(min_times[cols_idx[this.NAMES_EFF]].split('|')[2])
            this.maxfac_eff     = float(min_times[cols_idx[this.NAMES_EFF]].split('|')[3])

            while (line := f.readline()):
                if line.find("Western Washington Univ") != -1: 
                    continue
                for i,d in enumerate(line.split(',')):
                    if i != 1:
                        try:
                            cols[col_names[i]].append(float(d))
                        except:
                            cols[col_names[i]].append(0)
                    else:
                        cols[col_names[i]].append(d)

            this.raw_cols = cols
            
            this.events : dict[str, Event] = {}
            this.events[this.NAMES_COST]   = Event(cols[this.NAMES_COST],   lambda x: x)
            this.events[this.NAMES_DESIGN] = Event(cols[this.NAMES_DESIGN], lambda x: x)
            this.events[this.NAMES_PRES]   = Event(cols[this.NAMES_PRES],   lambda x: x)
            this.events[this.NAMES_ACCEL]  = Event(cols[this.NAMES_ACCEL],  lambda x: 95.5 * ((this.maxtime_accel/x) - 1)/((this.maxtime_accel/this.mintime_accel) - 1) + 4.5)
            this.events[this.NAMES_SKID]   = Event(cols[this.NAMES_SKID],   lambda x: 71.5 * ((this.maxtime_skid/x)**2 - 1)/((this.maxtime_skid/this.mintime_skid)**2 - 1) + 3.5)
            this.events[this.NAMES_AUTOX]  = Event(cols[this.NAMES_AUTOX],  lambda x: 118.5 * ((this.maxtime_autox/x) - 1)/((this.maxtime_autox/this.mintime_autox) - 1) + 6.5)
            this.events[this.NAMES_ENDURO] = Event(cols[this.NAMES_ENDURO], lambda x: 250 * ((this.maxtime_enduro/x) - 1)/((this.maxtime_enduro/this.mintime_enduro) - 1) + 25)
            this.events[this.NAMES_EFF]    = Event(cols[this.NAMES_EFF],    lambda x: 100 * ((x - this.minfac_eff) / (this.maxfac_eff - this.minfac_eff)))
            this.events[this.NAMES_TOTAL]  = Event(cols[this.NAMES_TOTAL],  lambda x: x)

    def EventPlacement(this, name : str, time : float):
        return this.events[name].get_placement(time)

    def EventScore(this, name : str, time : float):
        return this.events[name].get_score(time)
    
    def EfficiencyFactor(this, endurance_time, endurance_kw):
        avg_time = endurance_time / 22
        avg_kwh  = endurance_kw / 22
        avg_co2  = 0.65 * avg_kwh

        return (this.mintime_eff / avg_time) * (this.minco2_eff / avg_co2) 