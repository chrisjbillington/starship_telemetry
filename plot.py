import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pyatmos import coesa76

kmph = 1000 / 3600
km = 1000
s = 1
kPa = 1000

# Sidereal day
T_s = 86164.0905

# Gravitational constant
G = 6.67430e-11

# Earth mass
M = 5.97219e24

# Earth radius
R = 6_378_137.0

# Earth surface gravity
g = 9.8

# speed of sound at sea level
c0 = 343

# standard temperature at sea level
T0 = coesa76(0).T


STARBASE_LATITUDE = 25.99616 * np.pi / 180
STARBASE_LONGITUDE = -97.15444 * np.pi / 180
BEARING = 97.5 * np.pi / 180

# time between datapoints
dt = 1


def interp_between_steps(arr, assume_truncated=False):
    """Replace contents of arr with values interpolated from the points where arr
    changes in value."""
    x_midpoints = []
    y_midpoints = []
    for i, (left, right) in enumerate(zip(arr[:-1], arr[1:])):
        if left != right:
            x_midpoints.append(i + 0.5)
            if assume_truncated:
                y_midpoints.append(max(left, right))
            else:
                y_midpoints.append((left + right) / 2)


    interpolator = interp1d(
        x_midpoints,
        y_midpoints,
        kind='quadratic',
        bounds_error=False,
        fill_value="extrapolate",
    )
    out = interpolator(np.arange(len(arr)))
    # plt.plot(arr)
    # plt.plot(out)
    # plt.show()
    return out


def load_data():
    df = pd.read_csv('telemetry.csv')
    t = np.array(df['t'])
    v = np.array(df['v1'])
    y = np.array(df['h1'])
    valid = ~(np.isnan(t) | np.isnan(v) | np.isnan(y))
    return (t[valid], v[valid] * kmph, y[valid] * km)


def great_cicle(lat0, lon0, bearing, delta):
    """Given an initial longitude, latitude and bearing, return the longitude and
    latitude after travelling an angular distance delta along a great circle. All angles
    in radians and bearing is defined clockwise with respect to due north.
    """
    lat = np.arcsin(
        np.sin(lat0) * np.cos(delta) + np.cos(lat0) * np.sin(delta) * np.cos(bearing)
    )
    lon = lon0 + np.arctan2(
        np.sin(bearing) * np.sin(delta) * np.cos(lat0),
        np.cos(delta) - np.sin(lat0) * np.sin(lat),
    )
    lon = (lon + np.pi) % (2 * np.pi) - np.pi  # shift to [-pi, pi]
    return lat, lon


# Load raw data:
t_raw, v_raw, y_raw = load_data()

# Replace altitude data and time with an interpolation between the points where we
# actually see change:
y_interp = interp_between_steps(y_raw)
t = interp_between_steps(t_raw, assume_truncated=True)

# Filter speed and altitude to remove spurious features. Note odd-numbered window length
# necessary to avoid an offset (ensure there is an exact middle)
v = savgol_filter(v_raw, window_length=9, polyorder=2)
y = savgol_filter(y_interp, window_length=49, polyorder=2)

# Compute y component of velocity using a similar smoothing:
vy = savgol_filter(y_interp, window_length=49, polyorder=2, deriv=1) / dt

# Compute x component of velocity. Whilst vx should never be greater than v, noise makes
# it sometimes come out to be so. We allow it and use the negative square root of the
# absolute value in this case. That way, the noise will be more symmetric and average
# out better than if we clipped vx to always be less than v.
vx = np.sign(v**2 - vy**2) * np.sqrt(np.abs(v**2 - vy**2))

# Acceleration in x and y
ax = np.gradient(vx) / dt
ay = np.gradient(vy) / dt
a = np.gradient(v) / dt


# Integrate up downrange velocity to obtain downrange position
x = cumulative_trapezoid(vx, t, initial=0)

# Density of air as a function of altitude
rho = coesa76(y / km).rho

# Dynamic pressure (q, Pa)
q = 1 / 2 * rho * v**2

# Air temperature as a function of altitude
T = coesa76(y / km).T

# Speed of sound as a function of altitude
c = c0 * np.sqrt(T / T0)

# mach number
Ma = v / c

# Integrate up angular velocity to obtain the angular distance around the earth we have
# travelled
delta = cumulative_trapezoid(vx / (y + R), t, initial=0)

# Find the latitude and longitude of the trajectory
lat, lon = great_cicle(STARBASE_LATITUDE, STARBASE_LONGITUDE, BEARING, delta)

# find the azimuth (direction of movement, anticlockwise relative to due east) along the
# trajectory:
azimuth = np.arctan2(np.gradient(lat), np.gradient(lon))

# split downrange velocity into east-west and north-south components
v_ew = vx * np.cos(azimuth)
v_ns = vx * np.sin(azimuth)

# Transform into the nonrotating frame by adding the tangential velocity of the earth at
# the current altitude to the east-west component of velocity
v_ew += 2 * np.pi * (R + y) / T_s * np.cos(lat)

# sum back together to get the tangential component of velocity in the inertial frame:
vt = np.sqrt(v_ew**2 + v_ns**2)

# specific orbital energy:
epsilon = (vt**2 + vy**2) / 2 - G * M / (R + y)

# specific angular momentum:
h = (R + y) * vt

# semimajor axis:
r_semimajor = -G * M / (2 * epsilon)

# eccentricity
eccentricity = np.sqrt(1 + (2 * epsilon * h**2) / (G**2 * M**2))

# apogee and perigee
ra = r_semimajor * (1 + eccentricity)
rp = r_semimajor * (1 - eccentricity)


plt.figure(figsize=(10, 12))
plt.subplot(421)
plt.title("Raw speed and altitude data")
plt.plot(t, v_raw / kmph, color='k')
plt.ylabel("Speed (kmph)", color="k")
plt.tick_params(axis='y', labelcolor="k")
plt.xlabel("t (s)")
plt.twinx()
plt.plot(t, y_raw / km, color='C3')
# plt.plot(t, y_interp / km, color='k', linestyle=":")
plt.ylabel("Altitude (km)", color="C3")
plt.tick_params(axis='y', labelcolor="C3")

plt.subplot(422)
plt.title("x and y velocity")
plt.plot(t, vx / kmph, color='C0')
plt.ylabel("$v_x$ (kmph)", color="C0")
plt.tick_params(axis='y', labelcolor="C0")
plt.xlabel("t (s)")
plt.twinx()
plt.plot(t, vy / kmph, color='C3')
plt.ylabel("$v_y$ (kmph)", color="C3")
plt.tick_params(axis='y', labelcolor="C3")

plt.subplot(423)
plt.title("x and y position")
plt.plot(t, x / km, color='C0')
plt.ylabel("$x$ (km)", color="C0")
plt.tick_params(axis='y', labelcolor="C0")
plt.xlabel("t (s)")
plt.twinx()
plt.plot(t, y / km, color='C3')
plt.ylabel("$y$ (km)", color="C3")
plt.tick_params(axis='y', labelcolor="C3")

plt.subplot(424)
plt.title("Trajectory")
plt.plot(x / km, y / km, color='k', label="Actual")
plt.xlabel("$x$ (km)")
plt.ylabel("$y$ (km)")

plt.subplot(425)
plt.title("Acceleration")
plt.plot(t, ax / g, color='C0', label="$a_x$")
plt.plot(t, ay / g, color='C3', label="$a_y$")
plt.ylabel("Acceleration (g)")
plt.legend()
plt.xlabel("t (s)")

plt.subplot(426)
plt.title("Dynamic pressure and Mach number")
plt.plot(t, q / kPa, color="k")
plt.ylabel("q (kPa)")
plt.xlabel("t (s)")
plt.twinx()
plt.plot(t, Ma, color="C2")
plt.ylabel("Ma", color="C2")
plt.tick_params(axis='y', labelcolor="C2")
plt.tight_layout()

plt.subplot(427)
plt.title("Orbital radius")
plt.plot(t, r_semimajor / R, label="semimajor axis")
plt.plot(t, ra / R, label="apogee")
plt.plot(t, rp / R, label="perigee")
plt.ylabel("Orbital radius (Earth radii)")
plt.xlabel("t (s)")
plt.legend()

plt.subplot(428)
plt.title("Orbital altitude")
plt.plot(t, (r_semimajor - R) / km, label="semimajor axis")
plt.plot(t, (ra - R) / km, label="apogee")
plt.plot(t, (rp - R) / km, label="perigee")
plt.ylabel("Orbital altitude (km)")
plt.axis(ymin=-200, ymax=200)
plt.xlabel("t (s)")
plt.legend()

plt.tight_layout()

plt.savefig("telemetry.png")

plt.figure(figsize=(10, 5))
earth = Image.open("earth_8k.jpg")
plt.imshow(earth, extent=[-np.pi, np.pi, -np.pi / 2, np.pi / 2])
plt.plot([STARBASE_LONGITUDE], [STARBASE_LATITUDE], 'go')
plt.plot(lon, lat, 'g-')
plt.plot([lon[-1]], [lat[-1]], 'ro')
plt.axis(xmin=-np.pi, xmax=np.pi, ymin=-np.pi / 2, ymax=np.pi / 2)
plt.axhline(0, color='k')
plt.axis('off')
plt.subplots_adjust(0, 0, 1, 1)
plt.savefig("trajectory_map.png")

# And zoomed in on Texas
plt.axis(
    xmin=STARBASE_LONGITUDE - 5 * np.pi / 180,
    xmax=STARBASE_LONGITUDE + 5 * np.pi / 180,
    ymin=STARBASE_LATITUDE - 5 * np.pi / 180,
    ymax=STARBASE_LATITUDE + 5 * np.pi / 180,
)
plt.savefig("trajectory_map_zoomed.png")
plt.show()
