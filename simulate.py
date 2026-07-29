from track import trackDef, parse_csv, toTangentCurve, from_points
from vehicle import vehicle, polynomial, lookuptable_2D
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import odeint
from scipy.signal import resample, argrelextrema
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

    def run_endurance(self, vehicledata):
        pass

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
        d = parse_csv('res/data/eline.csv')
        points = list(zip(d['X'], d['Y']))
        fake_times = [i - d['I'][0] for i in d['I']]

        tr2 = from_points(points, list(map(lambda x : x * 1000, d['s'])))
        px, py = list(map(lambda x : x.p[0], tr2.segments)), list(map(lambda x : x.p[1], tr2.segments))
        cv = list(map(lambda x : x.c, tr2.segments))
        inflections = argrelextrema(np.array(d['V']), np.less)[0]
        def istwoaway(i, arr):
            return i in arr or i-1 in arr or i-2 in arr or i+1 in arr or i+2 in arr
        #inflection_arr = [d['V'][i] if istwoaway(i, inflections) and d['V'][i] < 14 else 1000 for i in range(0, len(d['V']))]
        inflection_arr = [100] * len(d['V'])
        mv = [min(vehicledata.max_velocity_of_c(x.c), v) for x,v in zip(tr2.segments, inflection_arr)]

        # Braking Zones
        last_v = mv[-1]
        last_s = tr2.segments[-1].s
        last_c = cv[-1]
        mvf = [0] * len(mv)
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

        # prefiltering
        last_v = mv[-1]
        v_a = 0
        for i in range(0, len(mv)):
            v = v_a + last_v
            dv = mv[i] - mv[i-1]

            if dv < 0 and v > mv[i]:
                v = mv[i]
                v_a = dv * 0.3
            else:
                v_a += math.copysign(8 / abs(v), mv[i] - v)
            mvf[i] = v
            last_v = v

        mvf = mv

        # Accel Zones
        v_final = [0] * len(mv)
        tq_req = [0] * len(mv)
        last_v = mv[-1]
        last_c = cv[-1]
        print('accel')
        power_used = 0
        ticks_accel = 0
        time = 0
        times = []
        dtimes = []
        powers = [0] * len(mv)
        accels = [0] * len(mv)
        
        for i in range(0, len(mv)):
            this_s = tr2.segments[i].s
            ds = abs(this_s - last_s)
            if ds > 100:
                ds = 4
            dt = abs(ds / last_v)
            time += dt
            times.append(time)
            dtimes.append(dt)

            a_c = (last_v * last_v) * abs(last_c)
            max_accel_y = vehicledata.max_lat_accel_g(last_v) * 9.806
            max_accel_a, tq_req[i] = vehicledata.max_accel_g(last_v, 100)
            max_accel_a = max_accel_a * 9.806 * math.sqrt(1 - (min(a_c / max_accel_y, 1))**2) # traction ellipse

            v_final[i] = abs(max_accel_a) * dt + last_v
            if v_final[i] > mvf[i]:
                v_final[i] = mvf[i]
            else:
                acceleration = max_accel_a
                accels[i] = max_accel_a
                Pow = vehicledata.mass * 2 * acceleration * ds * (1 / vehicledata.drivetrain_efficiency_at(acceleration, v_final[i]))
                ticks_accel += 1
                power_used += Pow
                powers[i] = Pow
            last_c = cv[i]
            last_v = v_final[i]
            last_s = this_s

        print('time:', time, 'energy:', power_used / 3600)

        plt.plot(np.linspace(0, 1000, len(mv)), mvf)
        plt.plot(np.linspace(0, 1000, len(mv)), v_final)
        plt.plot(np.linspace(0, 1000, len(mv)), [v for v in d['V']])
        plt.scatter([(1000/249) * i for i in inflections], [d['V'][i] for i in inflections])
        plt.show()

        return time * 22, power_used / 3600 * 22

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
        time = 0.05 # idk startup time or something
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
    car = vehicle(
                mass              = 280, 
                wheelbase         = 1.540, 
                trackwidth        = 1.175,
                cg_height         = 0.272,
                cg_bal            = 0.52, 
                # Tires
                coeff_fric_lon    = polynomial([1.40]), 
                coeff_fric_lat    = polynomial([1.35]), 
                wheel_radius      = 0.20, 
                # Drivetrain
                final_drive_ratio = 3.0, 
                pack_efficiency   = lookuptable_2D([[1,1],[1,1]]),
                motor_efficiency  = EMRAX228_Efficiency, 
                drive_efficiency  = 0.90, 
                max_regen_watts   = polynomial([5000]),
                # Aero
                drag_area         = 0.6, 
                downforce_area    = 0.0, 
                # High Voltage
                max_torque        = EMRAX228_MaxTorque,
                max_power_per_soc = polynomial([80000]), 
                capacity          = 5.8,
                # All wheel drive
                AWD=False
            )

    c_sc =  90 # assuming cost report isn't ignored
    d_sc = 100 # a reasonable design score if we get our shit together
    p_sc =  50 # a reasonable score ig
    # test
    #print_all_events(90.2, 18.8, 65, 4.860, 5.548, 50.077, 1602.185, 3.563, AllComp)
    Simulation.run_autocross_basic(car)

    vp, tp, tm, accel_time = Simulation.run_accel_basic(car, 75)
    print_event_results(CompetitionScores.NAMES_ACCEL, accel_time * 1.02, AllComp) 
    skidpad_time = Simulation.run_skidpad_basic(car, 9)
    print_event_results(CompetitionScores.NAMES_SKID, skidpad_time * 1.02, AllComp)

    print_all_events(c_sc, p_sc, d_sc, accel_time, skidpad_time, 48, 1470, 4.0, AllComp)

    xinterp,yinterp = np.meshgrid(np.linspace(0,6500,50), np.linspace(0,250,50))
    zinterp = np.fromfunction(lambda x, y : car.motor_efficiency.f(6500 * y/49, 250 * x/49), (50,50))

    fig, ax = plt.subplots()
    CS = ax.contour(xinterp,yinterp,zinterp, levels=[0.7,0.75,0.8,0.86,0.90,0.94,0.95,0.96])
    ax.clabel(CS, fontsize=10)
    ax.set_title('contours')

    plt.scatter(vp, tp, c=tm, cmap='prism')
    plt.colorbar()

    plt.show()
    #track = toTangentCurve(trackFromBezierCSV("defaulttrack.csv", 1.5))
    #Simulation.calculate_racing_line(track)
