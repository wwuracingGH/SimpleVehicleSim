# simple dynamic model of a vehicle

import numpy as np
import matplotlib.pyplot as plt

import math, matplotlib
AIRDENSITY = 1.204

class vehicle:
    def __init__(
            this,
            max_power           : float,
            max_torque          : float,
            mass                : float,
            coeff_fric_long     : float,
            coeff_fric_lat      : float,
            wheel_radius        : float,
            final_drive_ratio   : float,
            coeff_drag          : float,
            frontal_area        : float,
            coeff_downforce     : float
            ):
        this.max_power = max_power
        this.max_torque = max_torque
        this.wheel_radius = wheel_radius
        this.final_drive_ratio = final_drive_ratio

        this.mass = mass

        this.coeff_fric_lat = coeff_fric_lat
        this.coeff_fric_long = coeff_fric_long

        this.coeff_drag = coeff_drag
        this.frontal_area = frontal_area
        this.coeff_downforce = coeff_downforce

        this.tire_circumference = wheel_radius * 2 * np.radians(360)

        # END

    def max_accel_from_power(this, velocity : float):
        wheel_rps = velocity / this.tire_circumference
        motor_rps = wheel_rps * this.final_drive_ratio
        
        max_motor_torque = min(230, (this.max_power / (6.2831853 * motor_rps)))
        max_wheel_force = max_motor_torque * this.final_drive_ratio / this.wheel_radius

        print(motor_rps)
        if (motor_rps > 108): return 0 #if the motor rpm is greater than 6500
        else: return  max_wheel_force / this.mass / 9.81

    def downforce(this, velocity):
        return (this.mass + (this.coeff_downforce * (velocity**2)))

    #def tireForces(this)
    #    def lerp(a,b,t): (1-t) * a + t * b
    #    def invlerp(a,b,t): (t - a) / (b - a)

    #    slipx = -(1 + Rww/uw) #or, k is equal to -1 - speed of the tread over the surface / velocity of the wheel center
    #    slipa = atan(-vw/uw)  #or, a is equal to the angle between the wheels's velocity and the heading of the car

    #    coeff_fxmax = lerp(cf_x1, cf_x2, invlerp(Fz1, Fz2, Fz))
    #    coeff_fymax = lerp(cf_y1, cf_y2, invlerp(Fz1, Fz2, Fz))
    #    slipxmax    = lerp(cf_slipx1, cf_slipx2, invlerp(Fz1, Fz2, Fz))
    #    slipamax    = lerp(cf_slipa1, cf_slipa2, invlerp(Fz1, Fz2, Fz))

    #    slipxn = slipx/slipmax
    #    slipan = slipa/slipamax

    #    p = sqrt(slipxn**2 + slipan**2)

    #    ux = slipxmax*sin(Qx * arctan(pi*p/2 * arctan(Qx)))
    #    uv = slipymax*sin(Qy * arctan(pi*p/2 * arctan(Qy)))

    #    Fx = ux * Fz * slipxn/p
    #    Fy = uy * Fz * slipan/p


    def plot_elipse(this, velocity : float):
        t = np.linspace(0, 360, 360)
        
        x = this.coeff_fric_lat *  np.cos(np.radians(t)) * this.downforce(velocity) / this.mass
        y = this.coeff_fric_long * np.sin(np.radians(t)) * this.downforce(velocity) / this.mass 

        mv = this.max_accel_from_power(velocity)

        y = np.array(list(map((lambda n : min(n, mv)), y)))

        plt.plot(x,y, color=matplotlib.colors.hsv_to_rgb(((velocity / 60.0), 1, 1)))

if __name__ == '__main__':
    v = vehicle(80000, 230, 290, 1.4, 1.1, 0.22, 5, 0.35, 0.3, 0.3)
    
    plt.gca().set_aspect('equal')
    for i in range(0, 6):
        v.plot_elipse(float(i * 10))

    plt.title("Max accleration at speed")
    plt.show()
