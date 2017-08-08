/* This is my first app based on glup
 *  1. Input a polygon and display it.
 */
#include <geogram_gfx/glup_viewer/glup_viewer_gui.h>
#include <geogram_gfx/mesh/mesh_gfx.h>
#include <geogram_gfx/glup_viewer/glup_viewer_gui_private.h>
#include <geogram_gfx/glup_viewer/glup_viewer.h>
#include <geogram_gfx/third_party/ImGui/imgui.h>

#include <iostream>

namespace {
    using namespace GEO;

    class CVTOptGlupApplication : public GEO::Application {
    public:
        CVTOptGlupApplication(int argc, char** argv) :
            Application(argc, argv, "") {

            show_polygon_domain_ = true;
            attribute_min_ = 0.0f;
            attribute_max_ = 0.0f;
        }

        /**
         * \brief Draws the application menus.
         * \details This function overloads 
         *  Application::draw_application_menus(). It can be used to create
         *  additional menus in the main menu bar.
         */
        virtual void draw_application_menus() {
            if(ImGui::BeginMenu("Commands")) {
                if(ImGui::MenuItem("say hello")) {
                    // Command::set_current() creates the dialog box and
                    // invokes a function when the "apply" button of the
                    // dialog box is pushed.
                    //   It needs the prototype of the function as a string,
                    // to have argument names and default values.
                    //   If prototype is not specified, then default values are
                    // set to zero and argument names are arg0,arg1...argn.
                    //   The two other arguments are a pointer to an object and
                    // the pointer to the member function to be invoked (it is
                    // also possible to give a pointer to a global static
                    // function).
                    Command::set_current(
                        "say_hello(index_t nb_times=1)", 
                        this,
                        &CVTOptGlupApplication::say_hello
                    );
                }

                if(ImGui::MenuItem("compute")) {
                    // Another example of Command::set_current() with more
                    // information in the function prototype string: each
                    // argument and the function can have a tooltip text,
                    // specified between square brackets.
                    Command::set_current(
                        "compute("
                        "   index_t nb_iter=300 [number of iterations]"
                        ") [pretends to compute something]",
                        this,
                        &CVTOptGlupApplication::compute
                    );
                }
                ImGui::EndMenu();
            }
        }

        /**
         * \brief An example function invoked from a menu.
         * \see draw_application_menus()
         */
        void say_hello(index_t nb_times) {
            show_console();
            for(index_t i=1; i<=nb_times; ++i) {
                Logger::out("MyApp") << i << ": Hello, world" << std::endl;
            }
        }

        /**
         * \brief An example function invoked from a menu.
         * \see draw_application_menus()
         */
        void compute(index_t nb_iter) {
            
            // Create a progress bar
            ProgressTask progress("Computing", nb_iter);
            try {
                for(index_t i=0; i<nb_iter; ++i) {
                    // Insert code here to do the actual computation

                    // Update the progress bar.
                    progress.next();
                }
            } catch(TaskCanceled& ) {
                // This block is executed if the user pushes the "cancel"
                // button.
                show_console();
                Logger::out("Compute") << "Task was canceled by the user"
                                       << std::endl;
            }
        }

        virtual bool load(const std::string& filename){
            mesh_gfx_.set_mesh(nil);
            mesh_.clear(false, false);

            mesh_.vertices.set_dimension(2);
            mesh_.vertices.set_double_precision();

            index_t idx[] = {-1, -1, -1, -1};
            double *p = NULL;
            idx[0]  = mesh_.vertices.create_vertex();
            p = mesh_.vertices.point_ptr(idx[0]);
            p[0] = 0.0;
            p[1] = 0.0;

            idx[1] = mesh_.vertices.create_vertex();
            p = mesh_.vertices.point_ptr(idx[1]);
            p[0] = 1.0;
            p[1] = 0.0;

            idx[2] = mesh_.vertices.create_vertex();
            p = mesh_.vertices.point_ptr(idx[2]);
            p[0] = 1.0;
            p[1] = 1.0;

            idx[3] = mesh_.vertices.create_vertex();
            p = mesh_.vertices.point_ptr(idx[3]);
            p[0] = 0.0;
            p[1] = 1.0;
            std::cout << p[0] << p[1] << std::endl;

            mesh_.edges.create_edge(idx[0], idx[1]);
            mesh_.edges.create_edge(idx[1], idx[2]);
            mesh_.edges.create_edge(idx[2], idx[3]);
            mesh_.edges.create_edge(idx[3], idx[1]);
            mesh_gfx_.set_mesh(&mesh_);
            mesh_gfx_.set_animate(false);
            glup_viewer_set_region_of_interest(-0.1, -0.1, 0, 1.1, 1.1, 0);
            return true;
        }

        virtual void init_graphics() {
            init_colormaps();
            Application::init_graphics();
        }

        virtual void drow_scene() {
            if(mesh_gfx_.mesh() == nil){
                return;
            }
            glup_viewer_is_enabled(GLUP_VIEWER_IDLE_REDRAW);
            mesh_gfx_.unset_scalar_attribute();
            mesh_gfx_.set_mesh_color(1.0, 1.0, 1.0);
            mesh_gfx_.set_points_color(0.0, 1.0, 0.0);
            mesh_gfx_.set_points_size(1.0);
            mesh_gfx_.draw_vertices();
            mesh_gfx_.draw_edges();
            return;
        }
    protected:
        Mesh mesh_;
        MeshGfx mesh_gfx_;
        bool show_polygon_domain_;
        float attribute_min_;
        float attribute_max_;
    };
}

int main(int argc, char** argv) {
    CVTOptGlupApplication app(argc, argv);
    app.start();
    return 0;
}

