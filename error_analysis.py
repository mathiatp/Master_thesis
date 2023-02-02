import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator

to_percent = 100

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).

# radii = np.linspace(0.125, 1.0, n_radii)
# angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]

# ERROR THETA##
dist_min = 0.5
dist_max = 30
distance = np.linspace(dist_min,dist_max)
camer_h_min = 1
camer_h_max = 5
camera_h = np.linspace(camer_h_min,camer_h_max)
cam_h, dist = np.meshgrid(camera_h, distance)
delta_theta_deg = 1
delta_theta_rad = np.radians(delta_theta_deg)
error_theta = np.abs(-((cam_h**2+dist**2)/(cam_h*dist))*delta_theta_rad)

ax = plt.figure().add_subplot(projection='3d')

ax.plot_surface(dist, cam_h, error_theta, cmap=plt.cm.CMRmap, vmin = 0, vmax = 2.5, antialiased = False)
#TODO fix title x variable 
ax.set_title(r'$\eta_{\theta}(\mathbf{x}\check{})$ with $\Delta \theta$ = '+'{}'.format(delta_theta_deg) +r'$^\circ$')
ax.set_ylabel(r'Camera height, $h[m]$')
ax.set_xlabel(r'Distance, $d_w[m]$')
ax.set_zlabel(r'Distance error, $\eta_{\theta} [\%]$')
ax.set_zlim(0,2.5)
elevation_angle = 22
azimuthal_angle = 62
ax.view_init(elev=elevation_angle, azim=azimuthal_angle)
plt.savefig('Images/error_theta1_same_z.eps', format='eps')





## ERROR f##

# camera_F_height_vessel = 3.8
# # camera_F_height_vessel = 36
# distance_to_target = 15
# # distance_to_target = 100

# theta_min = np.radians(0)
# theta_max = np.radians(90)
# theta = np.linspace(theta_max,theta_min)

# f_min = 500
# f_max = 3000
# f = np.linspace(f_min,f_max)
# THETA, F= np.meshgrid(theta,f)
# delta_f = 1


# error_f = np.abs( ((np.sin(2* (np.arctan(camera_F_height_vessel/distance_to_target) - THETA) ))/(2*F)) 
#          * ((camera_F_height_vessel**2 + distance_to_target**2)/(camera_F_height_vessel*distance_to_target))* delta_f )*to_percent

# #TODO fix title/lables 
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot_surface(np.rad2deg(THETA), F, error_f, cmap=plt.cm.CMRmap,vmin = 0, vmax = 0.4, antialiased = False)

# ax.set_title(r'$\eta_{f}(\mathbf{x}\check{})$ with $\Delta f=$' +'{}'.format(delta_f) +'px.\n'
#                 +r'Distance = ' +'{}'.format(distance_to_target) +r'm. Camera height = '+'{}'.format(camera_F_height_vessel)+r'm.')
# ax.set_xlabel(r'$\theta[deg]$')
# ax.set_ylabel(r'$f[px]$')
# ax.invert_xaxis()

# ax.set_zlabel(r'Distance error, $\eta_{f}$  $[\%]$')

# elevation_angle = 18
# azimuthal_angle = 108
# ax.view_init(elev=elevation_angle, azim=azimuthal_angle)
# # plt.savefig('Images/error_f.eps', format='eps')






# ## ERROR y_pos##

# theta_min = np.radians(0)
# theta_max = np.radians(90)
# theta = np.linspace(theta_max,theta_min)

# f_min = 500
# f_max = 3000
# f = np.linspace(f_min,f_max)
# THETA, F= np.meshgrid(theta,f)
# delta_y = 3

# dist_y_pos = 15
# # dist_y_pos = 100
# camera_F_height_vessel = 3.8
# # camera_F_height_vessel = 36

# error_y_pos = np.abs( -(1 + np.cos(2*(np.arctan(camera_F_height_vessel/dist_y_pos ) - THETA ) ))/( 2 * F))\
#                 *((camera_F_height_vessel**2+dist_y_pos**2)/(camera_F_height_vessel*dist_y_pos))*delta_y*to_percent

# ax = plt.figure().add_subplot(projection='3d')

# ax.plot_surface(np.rad2deg(THETA),F, error_y_pos, cmap=plt.cm.CMRmap, vmin = 0, vmax = 2.5, antialiased = False)

# #TODO fix title x variable 
# ax.set_title(r'$\eta_{y_{pos}}(\mathbf{x}\check{})$ with $\Delta y_{pos}=$' + '{}'.format(delta_y)+r'px.'+'\n' 
#             +r'Distance = ' +'{}'.format(dist_y_pos)+r'm. Camera height = '+'{}'.format(camera_F_height_vessel)+r'm.')
# ax.set_xlabel(r'$\theta[deg]$')
# ax.set_ylabel(r'$f[px]$')
# ax.invert_xaxis()

# ax.set_zlabel(r'Distance error, $\eta_{y_{pos}} [\%]$')
# elevation_angle = 23
# azimuthal_angle = 127
# ax.view_init(elev=elevation_angle, azim=azimuthal_angle)
# # plt.savefig('Images/error_y_pos.eps', format='eps')


plt.show()