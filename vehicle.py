# simple dynamic model of a vehicle

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sc

import math, matplotlib
AIRDENSITY = 1.204
GRAV       = 9.806
REGEN_MIN_SPEED = 5 / 3.6 # rules don't allow regen below 5 km/h 
#source: trust me bro

def get_tir_coefs(file_path):
    tir_file = open(file_path,'r')
    lines = tir_file.readlines()

    coefs = {
        'FNOMIN': 4000,
        'LFZO': 1,
        'PDX1': 1.3,
        'PDX2': -0.15,
        'LMUX': 1,
        'PDY1': 1.1,
        'PDY2': -0.15,
        'LMUY': 1,
    }

    for line in lines:
        line = line.strip()
        if line and not line.startswith(('$', '!', '[')):
            param = line.split('=')[0].strip()
            value = line.split('=')[1].split('$')[0].strip()
            if param not in coefs:
                continue
            coefs[param] = float(value) # prob should put in try block later but all we need are floats

    tir_file.close()
    return coefs

class polynomial:
    def __init__(this, values : list[float]):
        this.values = values
        this.max_power = len(values)
    
    def f(this, x):
        n = this.values[-1]
        for i in range(1, this.max_power):
            n += this.values[- (i + 1)] * x ** i
        return n

def tir_to_mu(coefs, scale_lon=0.65, scale_lat=0.65): # 0.65 because ttc data is big
    LMUX = coefs['LMUX'] * scale_lon
    LMUY = coefs['LMUY'] * scale_lat

    fz0 = coefs['FNOMIN'] * coefs['LFZO']
    lon = polynomial([LMUX*coefs['PDX2']/fz0, LMUX*(coefs['PDX1'] - coefs['PDX2'])])
    lat = polynomial([LMUY*coefs['PDY2']/fz0, LMUY*(coefs['PDY1'] - coefs['PDY2'])])
    return lon, lat

class lookuptable_2D:
    def __init__(this, values : list[list[float]], xmin=0, xmax=1, ymin=0, ymax=1):
        this.spline = sc.RegularGridInterpolator((np.linspace(1,0,len(values[0])), np.linspace(1,0,len(values))), np.transpose(values), method='linear') 
        this.xmin, this.xmax, this.ymin, this.ymax = xmin, xmax, ymin, ymax

    # basic bilinear interpolation
    def f(this, x, y):
        xv = (x - this.xmin) / (this.xmax - this.xmin)
        yv = (y - this.ymin) / (this.ymax - this.ymin)
        return this.spline((yv,xv))

class vehicle:
    def __init__(
            this,
            # Basic
            mass                : float,
            wheelbase           : float,
            trackwidth          : float,
            tir_file_path       : string,
            #coeff_fric_lon      : polynomial, # coeff per load N
            #coeff_fric_lat      : polynomial, # coeff per load N
            wheel_radius        : float,
            cg_height           : float,
            cg_bal              : float,      # often called 'a'
            # Drivetrain
            final_drive_ratio   : float,
            pack_efficiency     : lookuptable_2D,
            motor_efficiency    : lookuptable_2D, # part of motor efficiency per torque
            drive_efficiency    : float,
            max_regen_watts     : polynomial,     # state of charge lookup table basically
            # Aero
            drag_area           : float,
            downforce_area      : float,
            # High Voltage
            max_torque          : float,
            max_power_per_soc   : polynomial,
            capacity            : float,
            # Misc
            AWD                 : bool = False,
            rolling_resistance  : float = 0.015  # Crr
        ):
        this.mass = mass
        this.wheelbase = wheelbase
        this.trackwidth = trackwidth

        coefs = get_tir_coefs(tir_file_path)
        this.coeff_fric_lon, this.coeff_fric_lat = tir_to_mu(coefs)
        #this.coeff_fric_lon = coeff_fric_lon
        #this.coeff_fric_lat = coeff_fric_lat
        this.wheel_radius = wheel_radius

        this.cg_height = cg_height
        this.cg_bal = cg_bal

        this.final_drive_ratio = final_drive_ratio
        this.pack_efficiency  = pack_efficiency
        this.motor_efficiency = motor_efficiency
        this.drive_efficiency = drive_efficiency
        this.max_regen_watts  = max_regen_watts

        this.drag_area = drag_area
        this.downforce_area = downforce_area

        this.max_torque = max_torque
        this.max_power_per_soc = max_power_per_soc
        this.capacity = capacity

        this.tire_circumference = wheel_radius * np.radians(360)
        this.max_speed = 6500 / (60 * final_drive_ratio) * this.tire_circumference

        this.AWD = AWD
        this.rolling_resistance = rolling_resistance

        # END

    def max_accel_g(this, velocity : float, soc : float, power_cap : float = None):
        motor_rps = (velocity / this.tire_circumference) * this.final_drive_ratio
        if motor_rps * 60 > 6500: return 0
        
        d_force = this.get_downforce(velocity) + (this.mass * GRAV)

        i = 0
        g_t = 1.4
        max_tire_g = (g_t * this.cg_bal) / (1 - (g_t * this.cg_height / this.wheelbase))
        while(i < 10):
            lon_pos = (max_tire_g * this.cg_height / this.wheelbase) + this.cg_bal
            lon_neg = 1 - lon_pos

            normal_force_rr = lon_neg * d_force

            g_t = this.coeff_fric_lon.f(normal_force_rr / 2)
            g_t2 = this.coeff_fric_lon.f(normal_force_rr / 2)
            max_tire_g = (g_t * this.cg_bal) / (1 - (g_t * this.cg_height / this.wheelbase))
            if (this.AWD): 
                max_tire_g += (g_t2 * (1 - this.cg_bal)) / (1 + (g_t2 * this.cg_height / this.wheelbase))
            max_tire_force = max_tire_g * (this.mass * GRAV)
            i += 1

        max_tire_g = max_tire_force / (this.mass * GRAV)
        max_tire_torque = max_tire_force * this.wheel_radius / this.final_drive_ratio
        battery_power = this.max_power_per_soc.f(soc)
        if power_cap is not None:
            battery_power = min(battery_power, power_cap)
        max_battery_torque = (battery_power * 9.54929677 / (motor_rps * 60))
        max_motor_torque = min(this.max_torque, max_battery_torque) * this.drive_efficiency * this.motor_efficiency.f(motor_rps * 60, min(min(max_battery_torque, max_tire_torque), this.max_torque))

        max_motor_force = max_motor_torque * this.final_drive_ratio / this.wheel_radius
        max_motor_force -= this.get_drag(velocity)
        max_motor_g = max_motor_force / (this.mass * GRAV)

        return min(max_motor_g, max_tire_g), min(max_motor_torque, max_tire_torque)


    def max_lat_accel_g(this, velocity : float):
        mu = this.coeff_fric_lat.values[-1]
        sens = 0
        if len(this.coeff_fric_lat.values) == 2:
            sens = this.coeff_fric_lat.values[0]

        if sens == 0:
            return (mu * this.mass * GRAV + 2 * mu * this.get_downforce(velocity)) / (this.mass * GRAV)

        msg = this.mass * GRAV * sens
        b = this.trackwidth
        hsq = this.cg_height * this.cg_height
        cv2 = this.get_downforce(velocity)

        root = (b * b) - (16 * hsq * sens * cv2 * (mu + sens * cv2 + msg)) - (4 * hsq * msg * (msg + (2 * mu)))
        root = (b - math.sqrt(root)) * b
        div = 4 * this.mass * GRAV * sens * hsq # for some reason grav wasnt here?

        return root / div

    def max_velocity_of_c(this, curvature):
        c = max(this.coeff_fric_lat.values[-1] * GRAV/(this.max_speed**2), abs(curvature))

        radius = 1 / c

        velocity = math.sqrt(this.max_lat_accel_g(0) * GRAV * radius)
        for _ in range(0, 10):
            velocity = math.sqrt(this.max_lat_accel_g(velocity) * GRAV * radius)

        return velocity 

    def max_braking_accel_g(this, velocity):
        mu = this.coeff_fric_lon.values[-1]
        sens = 0
        if len(this.coeff_fric_lon.values) == 2:
            sens = this.coeff_fric_lon.values[0]

        if sens == 0:
            return (mu * this.mass * GRAV + 2 * mu * this.get_downforce(velocity)) / (this.mass * GRAV)

        msg = this.mass * GRAV * sens
        b = this.wheelbase
        hsq = this.cg_height * this.cg_height
        cv2 = this.get_downforce(velocity)

        root = (b * b) - (16 * hsq * sens * cv2 * (mu + sens * cv2 + msg)) - (4 * hsq * msg * (msg + (2 * mu)))
        root = (b - math.sqrt(root)) * b
        div = 4 * this.mass * GRAV * sens * hsq

        return root / div

    def rpm_from_velocity(this, velocity):
        wheel_rps = velocity / (this.wheel_radius * 2 * math.pi)
        return wheel_rps * 60 * this.final_drive_ratio

    def drivetrain_efficiency_at(this, acceleration, velocity):
        rpm = this.rpm_from_velocity(velocity)
        f_a = (acceleration * this.mass) * (1 / this.drive_efficiency) 
        torque = (f_a / this.final_drive_ratio) * this.wheel_radius
        m_e = this.motor_efficiency.f(rpm, torque)
        return m_e * this.drive_efficiency 

    def get_drag(this, velocity):
        return (this.drag_area * (velocity**2) * 1.2)

    def get_rolling_resistance(this, velocity):
        return this.rolling_resistance * (this.mass * GRAV + this.get_downforce(velocity))

    def motor_efficiency_at(this, rpm, torque):
        me = this.motor_efficiency
        return float(me.f(min(max(rpm, me.xmin), me.xmax), min(max(torque, me.ymin), me.ymax)))

    # power drawn from the pack to put force down at velocity
    def battery_power(this, force, velocity, soc=100):
        rpm = this.rpm_from_velocity(velocity)
        if force >= 0:
            torque = force * this.wheel_radius / (this.final_drive_ratio * this.drive_efficiency)
            eff = this.drive_efficiency * this.motor_efficiency_at(rpm, torque)
            return force * velocity / eff

        if velocity < REGEN_MIN_SPEED:
            return 0
        # im capping it off of motor torque and recharge rate but idk if this is accurate
        regen_force = min(-force, this.max_torque * this.final_drive_ratio / this.wheel_radius)
        torque = regen_force * this.wheel_radius / this.final_drive_ratio
        eff = this.drive_efficiency * this.motor_efficiency_at(rpm, torque)
        return -min(regen_force * velocity * eff, this.max_regen_watts.f(soc))

    def get_tire_f(this, normal_force):
        return this.coeff_fric_lon  

    def get_downforce(this, velocity):
        return (this.downforce_area * (velocity**2) * AIRDENSITY)

    def plot_elipse(this, ax):
        t = np.linspace(0, 360, 60)
        velocity = np.linspace(0.001, 6500 / (60 * this.final_drive_ratio) * this.tire_circumference, 20)

        df_mult = ((this.mass * GRAV) + this.get_downforce(velocity)) / (this.mass * GRAV)
        
        xvals = []
        yvals = []
        zvals = []

        for v in velocity:
            x = this.max_lat_accel_g(v) * np.cos(np.radians(t))
            y = this.max_braking_accel_g(v) * np.sin(np.radians(t))
            mv = this.max_accel_g(v, 100)
            y = np.array(list(map((lambda n : min(n, mv)), y)))
            xvals.append(x)
            yvals.append(y)
            zvals.append([v] * len(x))

        ax.scatter(xvals,yvals,zvals)

    def plot_powersurface(this, ax):
        soc = np.linspace(5, 100, 20)
        velocity = np.linspace(0.001, 6500 / (60 * this.final_drive_ratio) * this.tire_circumference, 250)
        
        x, y = np.meshgrid(velocity, soc)
        z = np.array([[this.max_accel_g(v,c) for v in velocity] for c in soc])

        ax.set_xlabel("speed (m/s)")
        ax.set_ylabel("state of charge (%)")
        ax.set_zlabel("accel (G)")
        ax.plot_surface(x, y, z, rstride=20, cstride=10, cmap=matplotlib.cm.coolwarm)

if __name__ == '__main__':
    v = VEHICLE_V67 

    x = np.linspace(0, v.max_speed, 30)
    y = [v.max_lat_accel_g(xh) for xh in x]
    plt.plot(x, y)
    plt.show()

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
   
    #v.plot_elipse(ax)

    #plt.title("Max accleration at speed and SOC")
    #plt.show()
