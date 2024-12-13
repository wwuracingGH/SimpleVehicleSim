from track import trackDef, trackFromBezierCSV, normalize
from vehicle import vehicle, polynomial

import matplotlib.pyplot as plt
import matplotlib

import math
from time import sleep

#heat
colors = [
    0,
    88, #red, grey (coolish)
    196, #red
    208, #orange
    195, #blueish
    225, #purpleish
    244, #grey
    238, #grey
]

frames = [
    ([
        '          ',
        '          ',
        '    ▓▓    ',
        '          ',
        '          '
    ],
    [
        '          ',
        '          ',
        '    55    ',
        '          ',
        '          '     
    ], 100),
    ([
        '          ',
        '    ▄▄    ',
        '   █▓▓█   ',
        '    ▀▀    ',
        '          '
    ],
    [
        '          ',
        '    44    ',
        '   4444   ',
        '    44    ',
        '          '     
    ], 60),
    ([
        '          ',
        '   ▄▀▀▄   ',
        '  █ ▓▓ █  ',
        '   ▀▄▄▀   ',
        '          '
    ],
    [
        '          ',
        '   5555   ',
        '  553355  ',
        '   5555   ',
        '          '     
    ], 60),
    ([
        '    ▄▄    ',
        '  ▄▀  ▀▄  ',
        ' █  ▓▓  █ ',
        '  ▀▄  ▄▀  ',
        '    ▀▀    '
    ],
    [
        '    66    ',
        '  66  66  ',
        ' 6  33  6 ',
        '  66  66  ',
        '    66    '     
    ], 60),
    ([
        '   ▄▀▀▄   ',
        ' ▄▀ ▄▄ ▀▄ ',
        '█  █▓▓█  █',
        ' ▀▄ ▀▀ ▄▀ ',
        '   ▀▄▄▀   '
    ],
    [
        '   7777   ',
        ' 77 33 77 ',
        '7  3223  7',
        ' 77 33 77 ',
        '   7777   '
    ], 60),
    ([
        '          ',
        '    ▄▄    ',
        '   █▓▓█   ',
        '    ▀▀    ',
        '          '
    ],
    [
        '          ',
        '    33    ',
        '   3223   ',
        '    33    ',
        '          '
    ], 60),
    ([
        '          ',
        '    ▓▓    ',
        '  ▓▓▓▓▓▓  ',
        '    ▓▓    ',
        '          '
    ],
    [
        '          ',
        '    22    ',
        '  221122  ',
        '    22    ',
        '          '
    ], 60),
    ([
        '          ',
        '   ▓▓▓▓   ',
        '  ▓▓▓▓▓▓  ',
        '   ▓▓▓▓   ',
        '          '
    ],
    [
        '          ',
        '   2111   ',
        '  112211  ',
        '   1211   ',
        '          '
    ], 60),
    ([
        '          ',
        '   ▓▓▓▓   ',
        '  ▓▓▓▓▓▓  ',
        '   ▓▓▓▓   ',
        '          '
    ],
    [
        '          ',
        '   1122   ',
        '  111111  ',
        '   1211   ',
        '          '
    ], 60),
    ([
        '          ',
        '   ▓▓▓▓   ',
        '  ▓▓▓▓▓▓  ',
        '   ▓▓▓▓   ',
        '          '
    ],
    [
        '          ',
        '   1112   ',
        '  710711  ',
        '   1111   ',
        '          '
    ], 60),
    ([
        '          ',
        '   ▓▓▓▓   ',
        '  ▓▓▓▓▓▓  ',
        '   ▓▓▓▓   ',
        '          '
    ],
    [
        '          ',
        '   1661   ',
        '  166661  ',
        '   1661   ',
        '          '
    ], 60),
    ([
        '          ',
        '   ▓▓▓▓   ',
        '  ▓▓▓▓▓▓  ',
        '   ▓▓▓▓   ',
        '          '
    ],
    [
        '          ',
        '   0000   ',
        '  000000  ',
        '   0000   ',
        '          '
    ], 60),
    ([
        '          ',
        '   ▓▓▓▓   ',
        '  ▓▓▓▓▓▓  ',
        '   ▓▓▓▓   ',
        '          '
    ],
    [
        '          ',
        '   0000   ',
        '  007700  ',
        '   0000   ',
        '          '
    ], 60),
    ([
        '          ',
        '   ▓▓▓▓   ',
        '  ▓▓▓▓▓▓  ',
        '   ▓▓▓▓   ',
        '          '
    ],
    [
        '          ',
        '   0      ',
        '  0 77 0  ',
        '     00   ',
        '          '
    ], 60),
    ([
        '          ',
        '   ▓▓▓▓   ',
        '  ▓▓▓▓▓▓  ',
        '   ▓▓▓▓   ',
        '          '
    ],
    [
        '          ',
        '          ',
        '    77    ',
        '          ',
        '          '
    ], 60),
]

def explode(offset):
    offset_spaces = ' ' * offset

    for lines, cols, frametime in frames:
        print("\033[4F")
        computed_lines = [offset_spaces, offset_spaces, offset_spaces, offset_spaces, offset_spaces]
        for y in range(0, len(lines)):
            for x in range(0, len(lines[y])): 
                if cols[y][x] != ' ':
                    computed_lines[y] += f"\033[38;5;{colors[int(cols[y][x])]}m"+ lines[y][x]
                else:
                    computed_lines[y] += " "
            print(computed_lines[y])
        
        sleep(frametime / 1000.0)
        print("\033[3F")

    print("\033[0m")

class Simulation:
    def change_in_s(u, v, xi, curve, n):
        return (u * math.sin(xi) - v * math.cos(xi)) / (1 - n*curve)

    # S subscript f = 1/change_in_s
        
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

                        #    FL      FR      RL      RR
        self.tireforces = [(0, 0), (0, 0), (0, 0), (0, 0)]

        self.velocity      = (0, 0) # u, v in the paper (forward, right)
        self.vehicleyawrate     = 0 # ω (omega) in the paper, dψ
    
    @staticmethod
    def run_accel(vehicledata : vehicle, length : float):
        time = 0
        dt = 0.0166
        velocity = 0.01
        distance = 0

        display_travel_length = 100
        space_per_m = float(display_travel_length) / length
        spaces = 0

        print("\n")
 
        print(" /| __" + " " * 94 + "|")
        print("⌾════⌾" + " " * 94 + "|")       
        while distance < length:
            time += dt
            velocity += vehicledata.max_accel_g(velocity, 70) * 9.81 * dt
            distance += velocity * dt



            if spaces < int(space_per_m * distance):
                spaces = int(space_per_m * distance)

                print("\033[2F")
                print("\033[2F")
                print("\033[2F")
                print("\033[2F")
                
                toprint = " " * spaces
                
                print(vehicledata.max_accel_g(velocity, 70))
                print(str(float(int(time * 1000)) / 1000) + ": " + str(float(int(velocity * 100)) / 100) + "m/s")

                print(toprint + " /| __")
                print(toprint + "⌾════⌾")
            
            sleep(dt)

        
        print("YOU COMPLETED ACCEL IN " + str(float(int(time * 1000)) / 1000) + " SECONDS")
        print(str(float(int(velocity * 1000)) / 1000)) 

if __name__ == '__main__':
    car = vehicle(280, 2, 1.5, polynomial([-0.0001, 1.5]), polynomial([1.25]), 0.22,
                polynomial([-0.125, 0.50]), polynomial([0.5]), 
                3.3, polynomial([0.94]), 0.98, 
                0.0, 0.0, 
                230, polynomial([80000]), 6.2, False#polynomial([7.01587e-7, 0, 0.0793916, -19.2291, 1525.91552, 0]), 6.2
                )
    
    Simulation.run_accel(car, 75)

