INTRODUCTION TO MY COMP 643 PROJECT
------------

The Administration Menu module displays the entire administrative menu tree
(and most local tasks) in a drop-down menu, providing administrators one- or
two-click access to most pages.  Other modules may also add menu links to the
menu using hook_admin_menu_output_alter().

 * For a full description of the module, visit the project page:
   https://www.drupal.org/project/admin_menu

 * To submit bug reports and feature suggestions, or track changes:
   https://www.drupal.org/project/issues/admin_menu

REQUIREMENTS
------------

This module requires the following modules:

 * [Views](https://www.drupal.org/project/views)
 * [Panels](https://www.drupal.org/project/panels)

CONFIGURATION
-------------
 
 * Configure the user permissions in Administration » People » Permissions:

   - Use the administration pages and help (System module)

     The top-level administration categories require this permission to be
     accessible. The administration menu will be empty unless this permission
     is granted.

   - Access administration menu

     Users with this permission will see the administration menu at the top of
     each page.

