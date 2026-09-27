from track import trackDef, parse_csv, toTangentCurve, from_points
from vehicle import vehicle, polynomial, lookuptable_2D
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import odeint
from scipy.signal import resample, argrelextrema, savgol_filter, butter, lfilter, filtfilt, gauss_spline
from scipy.ndimage import gaussian_filter1d
from Scores import CompetitionScores, print_event_results, print_all_events
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib

import math
from time import sleep

from common import *

Scores2023 = CompetitionScores('res/scores/Scores2023.csv', '2023')
Scores2024 = CompetitionScores('res/scores/Scores2024.csv', '2024')
Scores2025 = CompetitionScores('res/scores/Scores2025.csv', '2025')
Scores2026 = CompetitionScores('res/scores/Scores2026.csv', '2026')
AllComp = [Scores2023, Scores2024, Scores2025, Scores2026]

class Simulation:
    def change_in_s(u, v, xi, curve, n):
        return (u * math.sin(xi) - v * math.cos(xi)) / (1 - n*curve)

    def change_in_n(u, v, xi, curve, n):
        return u * math.sin(xi) - v * math.cos(xi)

    def change_in_angleabs(yawrate, curve, delta_s):
        return yawrate - curve * delta_s
    
    def change_in_u_over_t(mass, yawrate, v, force_u):
        return (mass*yawrate*v + force_u) / mass

    def change_in_v_over_t(mass, yawrate, u, force_v):
        return (mass*yawrate*u + force_v) / mass

    def force_u(steeringangle, tireforces, drag):
        cd, sd = cos(steeringangle), sin(steeringangle)
        Fflx, Ffly, Ffrx, Ffry, Frlx, Frly, Frrx, Frry = tireforces 

        return cd * (Ffrx + Fflx) - sd * (Ffry +  Ffly) + (Frrx + Frlx) + drag

    def force_v(steeringangle, tireforces): 
        cd, sd = cos(steeringangle), sin(steeringangle)
        Fflx, Ffly, Ffrx, Ffry, Frlx, Frly, Frrx, Frry = tireforces 

        return cd * (Ffry + Ffly) + sd * (Ffrx + Ffry) + (Frry, Frly)

    def change_in_u_over_s(ds, du_t):
        return (1.0/ds) * du_t

    def change_in_v_over_s(ds, dv_t):
        return (1.0/ds) * dv_t

    def change_in_w_over_s(ds, dw_t):
        return (1.0/ds) * dw_t

    @staticmethod
    def run_endurance(vehicledata : vehicle, start_soc, endurance_power : polynomial):
        d = parse_csv('res/data/eline.csv')
        points = list(zip(d['X'], d['Y']))
        fake_times = [i - d['I'][0] for i in d['I']]

        tr2 = from_points(points, list(map(lambda x : x * 1000, d['s'])))
        px, py = list(map(lambda x : x.p[0], tr2.segments)), list(map(lambda x : x.p[1], tr2.segments))
        cv = list(map(lambda x : x.c, tr2.segments)) 
        b, a = butter(5, 0.3, fs=1)
        mv = [vehicledata.max_velocity_of_c(x) for x in [x for x in d['c2']]]
        prefiltering = True
        # Braking Zones
        last_v = mv[-1]
        last_s = tr2.segments[-1].s
        last_c = cv[-1]
        for i in range(len(mv) - 2, -len(mv) - 1, -1):
            this_s = tr2.segments[i].s
            ds = abs(last_s - this_s)
            if ds > 100:
                ds = 4
            dt = abs(ds / last_v)

            a_c = (last_v * last_v) * abs(last_c)
            max_accel_b = vehicledata.max_braking_accel_g(last_v)
            max_accel_y = vehicledata.max_lat_accel_g(last_v) * 9.806
            max_accel_b = max_accel_b * 9.806 * math.sqrt(1 - (min(a_c / max_accel_y, 1))**2) # traction ellipse

            new_v = abs(max_accel_b) * dt + last_v
            mv[i] = min(new_v, mv[i])

            last_v = mv[i]
            last_s = this_s
            last_c = cv[i]

        mvf = mv
        # prefiltering
        if prefiltering:
            last_v = mv[-1]
            v_a = 0
            for i in range(0, len(mv)):
                v = v_a + last_v
                dv = mv[i] - mv[i-1]

                if dv < 0 and v > mv[i]:
                    v = mv[i]
                    v_a = -1# max(v_a * 0.8 + dv * 0.3,-1)
                else:
                    v_a += math.copysign(1, mv[i] - v)
                mvf[i] = v
                last_v = v
        plt.plot(np.linspace(0,1000,len(mv)), mvf)
        # Accel Zones
        v_final = [0] * len(mv)
        tq_req = [0] * len(mv)
        last_v = 4
        last_c = cv[-1]
        energy_used = 0
        time = 0
        soc = start_soc
        start_temp = 25
        delta_t_pack = 0
        last_s = 0
        for _lap in range(0, 22):
            print(f"lap {_lap}")
            input = [0] * len(mv)
            first_v = last_v
            for i in range(0, len(mv)):
                this_s = tr2.segments[i].s
                ds = abs(this_s - last_s)
                if ds > 100:
                    ds = 4
                dt = abs(ds / last_v)
                a_c = (last_v * last_v) * abs(last_c)
                max_accel_y = vehicledata.max_lat_accel_g(last_v) * 9.806
                max_accel_a, _ = vehicledata.max_accel_g(last_v, soc * 100, power_curve=endurance_power)
                max_accel_a = max_accel_a * 9.806
                max_accel_b = vehicledata.max_braking_accel_g(last_v) * 9.806

                v_m = abs(max_accel_a) * dt + last_v
                if v_m > mvf[i]:
                    v_m = mvf[i]
                    input[i] = (v_m - last_v) / (dt * max_accel_b)
                else:
                    power = max_accel_a * vehicledata.mass * v_m
                    is_trac = power < endurance_power.f(soc)
                    input[i] = math.sqrt(1 - (min(a_c / max_accel_y, 1))**2) if is_trac else 1

                last_c = cv[i]
                last_v = v_m
                last_s = this_s

            last_v = first_v

            input = gaussian_filter1d(input, 0.55)

            for i in range(0, len(mv)):
                this_s = tr2.segments[i].s
                ds = abs(this_s - last_s)
                if ds > 100:
                    ds = 4
                dt = abs(ds / (last_v))
                time += dt

                max_accel_a, tq_req[i] = vehicledata.max_accel_g(last_v, soc * 100, power_curve=endurance_power)
                max_accel_b = vehicledata.max_braking_accel_g(last_v) * 9.806
                max_accel_a *= 9.806

                if input[i] < 0:
                    v_final[i] = max((max_accel_b * dt * input[i]) + last_v, 5)
                    acceleration = max_accel_b * input[i]
                else:
                    v_final[i] = (max_accel_a * dt * input[i]) + last_v
                    acceleration = max_accel_a * input[i]

                energy, excess_energy = vehicledata.energy_used_at(acceleration, v_final[i], soc * 100, dt, start_temp + delta_t_pack)

                energy_used += energy
                delta_t_pack += excess_energy / 22440
                soc -= ((energy + excess_energy) / (vehicledata.capacity * 3600000))

                last_c = cv[i]
                last_v = v_final[i]
                last_s = this_s

            plt.plot(np.linspace(0, 1000, len(mv)), v_final)

        plt.plot(np.linspace(0, 1000, len(mv)), [v for v in d['V']])
        print(f'time:{time:.2f}\tenergy:{energy_used / 3600:.2f}\tsoc_end: {soc*100:.1f}\tchange in temp: {delta_t_pack:.2f}')
        plt.show()

        return time, energy_used / 3600000

    def __init__(self, vehicledata : vehicle, track : trackDef):
        self.vehicledata = vehicledata
        self.track = track

        # θ = track absolute angle, track.getAngle(s)
        # N = track.width
        # C = 1/R, track.getCurve(s)
        # t = track tangent vector, track.getNormal(s)

        self.centerlinedist     = 0 # s in the paper
        self.vehicleoffset      = 0 # n in the paper
        
        self.vehicleangletrack  = 0 # ξ (xi) in the paper 
        self.vehicleangleabs    = 0 # ψ (psi) in the paper
        self.vehiclesteering    = 0 # δ (delta) in the paper

                        #      FL         FR         RL         RR
        self.tireforces = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]

        self.velocity      = (0, 0) # u, v in the paper (forward, right)
        self.vehicleyawrate     = 0 # ω (omega) in the paper, dψ
       
        self.n_points = 80
        
        self.Sp_mat = np.linspace(0, track.length, 80)
        self.Sp_d = [track.length / 80] * 80
        self.Up_mat = [(0, 0, 0)] * 80 # control variable vector
        self.Xp_mat = []

    def run_autocross_basic(vehicledata : vehicle):
        d = parse_csv('res/data/aline.csv')
        fake_times = [i - d['I'][0] for i in d['I']]

        cv = [x for x in d['c2']]
        inflections = argrelextrema(np.array(d['V']), np.less)[0]
        b, a = butter(3, 0.25, fs=1)
        mv = [vehicledata.max_velocity_of_c(x) for x in cv] 
        prefiltering = False
        # Braking Zones
        last_v = mv[-1]
        last_s = d['s'][-1]
        last_c = cv[-1]
        for i in range(len(mv) - 2, -len(mv) - 1, -1):
            this_s = d['s'][i]
            ds = abs(last_s - this_s)
            if ds > 100:
                ds = 4
            dt = abs(ds / last_v)

            a_c = (last_v * last_v) * abs(last_c)
            max_accel_b = vehicledata.max_braking_accel_g(last_v)
            max_accel_y = vehicledata.max_lat_accel_g(last_v) * 9.806
            max_accel_b = max_accel_b * 9.806 * math.sqrt(1 - (min(a_c / max_accel_y, 1))**2) # traction ellipse

            new_v = abs(max_accel_b) * dt + last_v
            mv[i] = min(new_v, mv[i])

            last_v = mv[i]
            last_s = this_s
            last_c = cv[i]

        # Accel Zones
        last_v = 3
        last_c = cv[-1]
        power_used = 0
        time = 0
        last_s = 0
        soc = 0.9
        start_temp = 28.5
        delta_t_pack = 0
        energy_used = 0
        newdists = np.linspace(0,d['s'][-1],1000)
        mvf = np.interp(newdists, xp=d['s'], fp=mv)
        cv = np.interp(newdists, xp=d['s'], fp=cv)
        mvf[0] = 3
        mvf[1] = 4
        input = [0] * len(mvf)
        tq_req = [0] * len(mvf)
        v_final = [0] * len(mvf)
        for i in range(0, len(mvf)):
            this_s = newdists[i]
            ds = abs(this_s - last_s)
            if ds > 100:
                ds = 4
            dt = abs(ds / last_v)
            a_c = (last_v * last_v) * abs(last_c)
            max_accel_y = vehicledata.max_lat_accel_g(last_v) * 9.806
            max_accel_a, _ = vehicledata.max_accel_g(last_v, soc * 100)
            max_accel_a = max_accel_a * 9.806
            max_accel_b = vehicledata.max_braking_accel_g(last_v) * 9.806

            v_m = abs(max_accel_a) * dt + last_v
            if v_m > mvf[i]:
                v_m = mvf[i]
                input[i] = (v_m - last_v) / (dt * max_accel_b)
            else:
                power = max_accel_a * vehicledata.mass * v_m
                is_trac = power < vehicledata.max_power_per_soc.f(soc)
                input[i] = math.sqrt(1 - (min(a_c / max_accel_y, 1))**2) if is_trac else 1

            last_c = cv[i]
            last_v = v_m
            last_s = this_s

        last_v = 3
        last_s = 0
        input = gaussian_filter1d(input, 0.1)

        for i in range(0, len(mvf)):
            this_s = newdists[i]
            ds = abs(this_s - last_s)
            if ds > 100:
                ds = 4
            dt = abs(ds / (last_v))
            time += dt

            max_accel_a, tq_req[i] = vehicledata.max_accel_g(last_v, soc * 100)
            max_accel_b = vehicledata.max_braking_accel_g(last_v) * 9.806
            max_accel_a *= 9.806

            if input[i] < 0:
                v_final[i] = max((max_accel_b * dt * input[i]) + last_v, 0.0001)
                acceleration = max_accel_b * input[i]
            else:
                v_final[i] = (max_accel_a * dt * input[i]) + last_v
                acceleration = max_accel_a * input[i]

            #energy, excess_energy = vehicledata.energy_used_at(acceleration, v_final[i], soc * 100, dt, start_temp + delta_t_pack)

#            energy_used += energy
#            delta_t_pack += excess_energy / 22440
#            soc -= ((energy + excess_energy) / (vehicledata.capacity * 3600000))

#            print(dt, ds, v_final[i])
            last_c = cv[i]
            last_v = v_final[i]
            last_s = this_s

        delta_t_pack = 0
        test_energy = 0
        soc = 0.73
        as_ = []
        temp = 28.5
        for i in range(0, len(d['V'])):
            this_s = d['s'][i]
            ds = abs(this_s - last_s)
            if ds > 100:
                ds = 4
            dt = abs(ds / last_v)

            v = d['V'][i]
            a = (v - last_v) / dt
            as_.append(a)
            if a > 0:
                energy, excess_energy = vehicledata.energy_used_at(a, v, soc * 100, dt, temp + delta_t_pack)
                test_energy += energy
                delta_t_pack += excess_energy / 22440
                soc -= ((energy + excess_energy) / (vehicledata.capacity * 3600 * 1000))

            last_s = this_s
            last_v = v     

        print(f'test soc: {soc:.3f}\ttest energy: {test_energy/3600:.2f}\ttest_dT: {delta_t_pack}')
        print('time:', time, 'energy:', power_used / 3600)

        plt.plot(newdists, mvf)
        plt.plot(newdists, v_final)
        plt.plot(np.linspace(0, d['s'][-1], len(mv)), [v for v in d['V']])
        plt.scatter([(d['s'][-1]/249) * i for i in inflections], [d['V'][i] for i in inflections])
        plt.show()

        return time, power_used / 3600

    @staticmethod
    def run_skidpad_basic(vehicledata : vehicle, radius):
        v = 0
        for i in range(0, 10):
            ml_accel = vehicledata.max_lat_accel_g(v) * 9.81
            v = math.sqrt(ml_accel * radius)

        time = (radius * 2 * math.pi) / (v * 0.98)
        
        return time

    @staticmethod
    def run_accel_basic(vehicledata : vehicle, length : float):
        print('\n')
        time = 0.1 # idk startup time or something
        dt = 0.005
        velocity = 0.001
        distance = -0.3
        
        power_used = 0

        display_travel_length = 60
        space_per_m = float(display_travel_length) / length
        spaces = 0
        zto60time = -1

        vpoints = []
        tpoints = []
        times = []

        print("\n")
 
        print(" /| __" + " " * display_travel_length + "|")
        print("⌾════⌾" + " " * display_travel_length + "|")       
        while distance < length:
            acceleration, torque_req = vehicledata.max_accel_g(velocity, 100)
            acceleration *= 9.81
            if distance >= -0.0001:
                time += dt
            else: # some sort of arbitrary ramp ig
                acceleration *= 1/0.5 * (0.5 + distance)
                torque_req   *= 1/0.5 * (0.5 + distance)

            velocity += acceleration * dt
            distance += velocity * dt
            power_used += vehicledata.mass * acceleration * velocity * dt * (1 / vehicledata.drivetrain_efficiency_at(acceleration, velocity))

            if spaces < int(space_per_m * distance):
                spaces = int(space_per_m * distance)

                print("\033[2F")
                print("\033[2F")
                print("\033[2F")
                print("\033[2F")
                print("\033[2F")
 
                if (zto60time < 0):
                    if (velocity >= 26.8224):
                        zto60time = time
                else:
                    print("\033[2F")
                    print("zero to 60mph time: " + str(round(zto60time * 100) / 100))
               
                toprint = " " * spaces


                print("accel in g: " + str(acceleration / 9.806))
                print("torque req: " + str(torque_req))
                print(str(float(int(time * 1000)) / 1000) + ": " + str(float(int(velocity * 100)) / 100) + "m/s           ")
                print(toprint + " /| __")
                print(toprint + "⌾════⌾")

            sleep(dt / 2)

            vpoints.append(vehicledata.rpm_from_velocity(velocity))
            tpoints.append(torque_req)
            times.append(time)
        
        power_used /= 3600000
        print("POWER USED: " + str(float(round(power_used * 1000)) / 1000) + 'kWh')
        
        return vpoints, tpoints, times, time

if __name__ == '__main__':
    car = VEHICLE_V67

    c_sc =  90 # assuming cost report isn't ignored
    d_sc =  80 # a reasonable design score if we get our shit together
    p_sc =  40 # a reasonable score ig
    # test
    #print_all_events(90.2, 18.8, 65, 4.860, 5.548, 50.077, 1602.185, 3.563, AllComp)
    endurance_time, endurance_energy = Simulation.run_endurance(car, 0.95, test_endurance2)
    autocross_time, _ = Simulation.run_autocross_basic(car)

    vp, tp, tm, accel_time = Simulation.run_accel_basic(car, 75)
    print_event_results(CompetitionScores.NAMES_ACCEL, accel_time * 1.02, AllComp) 
    skidpad_time = Simulation.run_skidpad_basic(car, 9)
    print_event_results(CompetitionScores.NAMES_SKID, skidpad_time * 1.02, AllComp)

    print_all_events(c_sc, p_sc, d_sc, accel_time, skidpad_time, autocross_time, endurance_time, endurance_energy, AllComp)

    xinterp,yinterp = np.meshgrid(np.linspace(0,car.motor_efficiency.xmax,50), np.linspace(0,car.max_torque * (4 if car.AWD else 1),50))
    zinterp = np.fromfunction(lambda x, y : car.motor_efficiency.f(car.motor_efficiency.xmax * y/49, car.max_torque * x/49), (50,50))

    fig, ax = plt.subplots()
    CS = ax.contour(xinterp,yinterp,zinterp, levels=[0.7,0.75,0.8,0.86,0.90,0.94,0.95,0.96])
    ax.clabel(CS, fontsize=10)
    ax.set_title('contours')

    plt.scatter(vp, tp, c=tm, cmap='prism')
    plt.colorbar()

    plt.show()
    #track = toTangentCurve(trackFromBezierCSV("defaulttrack.csv", 1.5))
    #Simulation.calculate_racing_line(track)
