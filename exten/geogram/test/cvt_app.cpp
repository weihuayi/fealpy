
#include <geogram_gfx/glup_viewer/glup_viewer.h>
#include <geogram/voronoi/CVT.h>
#include <vector>

using namespace GEO;
class CVTApplication : public Application 
{
    public:

        CVTApplication( int argc, char** argv,
                const std::string& usage
                ) : Application(argc, argv, usage) 
        {

            m_show_domain = true;
            m_show_delaunay = true;
            m_show_rvd = true;
            m_show_points = true;

            m_lloyd_maxit = 10;
            m_newton_maxit = 100;

            create_domain();
            m_cvt = new CentroidalVoronoiTesselation(&m_domain);
            m_cvt->set_show_iterations(true);
            m_num_points = 10;
            init_points(m_num_points);


            // Key shortcuts.
            glup_viewer_add_toggle('d', &m_show_domain, "domain");
            glup_viewer_add_toggle('m', &m_show_delaunay,  "delaunay");            
            glup_viewer_add_toggle('r', &m_show_rvd, "rvd");
            glup_viewer_add_toggle('p', &m_show_points, "points");
            glup_viewer_add_key_func(
                ' ', &CVTApplication::one_lloyd_iteration,
                "one step of Lloyd iteration"
            );

            // Define the 3d region that we want to display
            // (xmin, ymin, zmin, xmax, ymax, zmax)
            glup_viewer_set_region_of_interest(
                    0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f
                    );
            glup_viewer_disable(GLUP_VIEWER_BACKGROUND);
            glup_viewer_disable(GLUP_VIEWER_3D);
        }

        virtual ~CVTApplication() 
        {
            delete m_cvt;
        }

        void init_points(index_t num_points)
        {
            m_cvt->compute_initial_sampling(num_points);
            m_cvt->compute_surface(&m_rvd_mesh);
            m_rvd = m_cvt->RVD();
            m_delaunay = m_rvd->delaunay();
            m_rvd->compute_RVD(m_rvd_mesh);
        }

        void create_domain()
        {
            m_domain.clear(false, false);

            m_domain.vertices.set_dimension(3);
            m_domain.vertices.set_double_precision();

            index_t idx[] = {-1, -1, -1, -1};
            double *p = NULL;
            idx[0]  = m_domain.vertices.create_vertex();
            p = m_domain.vertices.point_ptr(idx[0]);
            p[0] = 0.0;
            p[1] = 0.0;
            p[2] = 0.0;

            idx[1] = m_domain.vertices.create_vertex();
            p = m_domain.vertices.point_ptr(idx[1]);
            p[0] = 1.0;
            p[1] = 0.0;
            p[2] = 0.0;

            idx[2] = m_domain.vertices.create_vertex();
            p = m_domain.vertices.point_ptr(idx[2]);
            p[0] = 1.0;
            p[1] = 1.0;
            p[2] = 0.0;

            idx[3] = m_domain.vertices.create_vertex();
            p = m_domain.vertices.point_ptr(idx[3]);
            p[0] = 0.0;
            p[1] = 1.0;
            p[2] = 0.0;

            m_domain.facets.create_triangle(idx[1], idx[2], idx[0]);
            m_domain.facets.create_triangle(idx[3], idx[0], idx[2]);
            m_domain.facets.connect();
            m_domain.show_stats();

            return; 
        }

        /**
         * \brief Displays and handles the GUI for object properties.
         * \details Overloads Application::draw_object_properties().
         */
        virtual void draw_object_properties() 
        {
            ImGui::Checkbox("domain [d]", &m_show_domain);
            ImGui::Checkbox("delaunay [m]", &m_show_delaunay);
            ImGui::Checkbox("rvd [r]", &m_show_rvd);
            ImGui::Checkbox("points [p]", &m_show_points);
            ImGui::Separator();
            if(ImGui::Button("reset generators"))
            {
                init_points(m_num_points);
            }
            ImGui::InputInt("np", (int*)&m_num_points);

            ImGui::Separator();
            if(ImGui::Button("Lloyd iteration"))
            {
                m_cvt->Lloyd_iterations(m_lloyd_maxit);
                m_rvd->compute_RVD(m_rvd_mesh);
            }
            ImGui::InputInt("lloyd maxit", (int*)&m_lloyd_maxit);

            ImGui::Separator();
            if(ImGui::Button("Newton iteration"))
            {
                m_cvt->Newton_iterations(m_newton_maxit);
                m_rvd->compute_RVD(m_rvd_mesh);
            }
            ImGui::InputInt("newton maxit", (int*)&m_newton_maxit);
	    }

        /**
         * \brief Gets the instance.
         * \return a pointer to the current DemoGlupApplication.
         */
        static CVTApplication* instance() 
        {
            CVTApplication* result =
                dynamic_cast<CVTApplication*>(Application::instance());
            geo_assert(result != nil);
            return result;
        }

        /**
         * \brief Draws the scene according to currently set primitive and
         *  drawing modes.
         */
        virtual void draw_scene() 
        {
            glupSetColor3f(GLUP_FRONT_COLOR, 1.0f, 1.0f, 0.0f);
            glupSetColor3f(GLUP_BACK_COLOR, 1.0f, 0.0f, 1.0f);
            glupEnable(GLUP_LIGHTING);
            if(m_show_domain)
                draw_domain_border();

            if(m_show_delaunay)
                draw_delaunay();

            if(m_show_rvd)
            {
                draw_rvd_edges();
                draw_rvd_cells();
            }

            if(m_show_points)
                draw_points();
        }

        void draw_domain_border()
        {
            glupSetMeshWidth(4);
            glupBegin(GLUP_LINES);
            for(index_t i=0; i< m_domain.facets.nb(); ++i)
            {
                index_t nv = m_domain.facets.nb_vertices(i);
                for(index_t j=0; j < nv; ++j)
                {
                    index_t idx = m_domain.facets.adjacent(i, j);
                    if(idx == NO_FACET)
                    {
                        glupVertex(m_domain.vertices.point(m_domain.facets.vertex(i, j)));
                        glupVertex(m_domain.vertices.point(m_domain.facets.vertex(i, (j+1)%nv)));
                    }
                }
            }
            glupEnd();
        }

        void draw_points()
        {
            glupEnable(GLUP_LIGHTING);
            glupSetPointSize(GLfloat(10.0));
            glupDisable(GLUP_VERTEX_COLORS);
            glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 0.0f, 1.0f, 1.0f);
            glupBegin(GLUP_POINTS);
            for(index_t i=0; i< m_delaunay->nb_vertices(); ++i)
            {
                glupVertex3dv(m_delaunay->vertex_ptr(i));
            }
            glupEnd();
            glupDisable(GLUP_LIGHTING);
        }

        void draw_delaunay()
        {
            glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 1.0, 0.5, 1.0);
            glupSetMeshWidth(4);
            glupBegin(GLUP_LINES);
            for(index_t c=0; c < m_delaunay->nb_cells(); ++c) {
                const signed_index_t* cell = m_delaunay->cell_to_v() + 3*c;
                for(index_t e=0; e<3; ++e) {
                    signed_index_t v1 = cell[e];
                    signed_index_t v2 = cell[(e+1)%3];
                    glupVertex3dv(m_delaunay->vertex_ptr(index_t(v1)));
                    glupVertex3dv(m_delaunay->vertex_ptr(index_t(v2)));
                }
            }
            glupEnd();
        }

        void draw_rvd_edges()
        {
            glupSetColor3f(GLUP_FRONT_AND_BACK_COLOR, 0.0, 0.0, 0.0);
            glupSetMeshWidth(4);
            glupBegin(GLUP_LINES);
            index_t * region = (index_t *)m_rvd_mesh.facets.attributes().find_attribute_store("region")->data();
            for(index_t i=0; i< m_rvd_mesh.facets.nb(); ++i)
            {
                index_t nv = m_rvd_mesh.facets.nb_vertices(i);
                for(index_t j=0; j < nv; ++j)
                {
                    index_t idx = m_rvd_mesh.facets.adjacent(i, j);
                    if(idx == NO_FACET || region[idx] != region[i])
                    {
                        glupVertex(m_rvd_mesh.vertices.point(m_rvd_mesh.facets.vertex(i, j)));
                        glupVertex(m_rvd_mesh.vertices.point(m_rvd_mesh.facets.vertex(i, (j+1)%nv)));
                    }
                }
                
            }
            glupEnd();
        }

        void draw_rvd_cells()
        {
            glupEnable(GLUP_VERTEX_COLORS);
            glupBegin(GLUP_TRIANGLES);
            index_t * region = (index_t *)m_rvd_mesh.facets.attributes().find_attribute_store("region")->data();
            for(index_t i=0; i< m_rvd_mesh.facets.nb(); ++i)
            {
                index_t nv = m_rvd_mesh.facets.nb_vertices(i);
                glup_viewer_random_color_from_index(int(region[i]));
                for(index_t j=0; j < nv; ++j)
                {
                    glupVertex3dv(m_delaunay->vertex_ptr(region[i]));
                    glupVertex(m_rvd_mesh.vertices.point(m_rvd_mesh.facets.vertex(i, j)));
                    glupVertex(m_rvd_mesh.vertices.point(m_rvd_mesh.facets.vertex(i, (j+1)%nv)));
                }
                
            }
            glupEnd();
            glupDisable(GLUP_VERTEX_COLORS);  
        }

        virtual void init_graphics() 
        {
            Application::init_graphics();
        }

        /**
         * \brief Draws the application menus.
         * \details This function overloads 
         *  Application::draw_application_menus(). It can be used to create
         *  additional menus in the main menu bar.
         */
        virtual void draw_application_menus() 
        {
            if(ImGui::BeginMenu("Commands")) 
            {
                if(ImGui::MenuItem("say hello")) 
                {
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
                        &CVTApplication::say_hello
                    );
                }

                if(ImGui::MenuItem("compute")) 
                {
                    // Another example of Command::set_current() with more
                    // information in the function prototype string: each
                    // argument and the function can have a tooltip text,
                    // specified between square brackets.
                    Command::set_current(
                        "compute("
                        "   index_t nb_iter=300 [number of iterations]"
                        ") [Lloyd iteration]",
                        this,
                        &CVTApplication::compute
                    );
                }

//                if(ImGui::MenuItem("reinit generators"))
//                {
//                    Command::set_current(
//                        "reinit generators("
//                        "   index_t num_points =300 [number of generators]"
//                        ") [reinit the generators]",
//                        this,
//                        &CVTApplication::init_points
//                        );
//                }
                ImGui::EndMenu();
            }
        }

        /**
         * \brief An example function invoked from a menu.
         * \see draw_application_menus()
         */
        void say_hello(index_t nb_times) 
        {
            show_console();
            for(index_t i=1; i<=nb_times; ++i) {
                Logger::out("MyApp") << i << ": Hello, world" << std::endl;
            }
        }

        /**
         * \brief An example function invoked from a menu.
         * \see draw_application_menus()
         */
        void compute(index_t nb_iter) 
        {
            
            // Create a progress bar
            ProgressTask progress("Computing", nb_iter);
            try {
                for(index_t i=0; i<nb_iter; ++i) 
                {
                    // Insert code here to do the actual computation
                    m_cvt->Lloyd_iterations(1);
                    m_rvd->compute_RVD(m_rvd_mesh);

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


        static void one_lloyd_iteration()
        {
            instance()->m_cvt->Lloyd_iterations(1);
            instance()->m_rvd->compute_RVD(instance()->m_rvd_mesh);
        }



    private:
        bool m_show_domain;
        bool m_show_delaunay;
        bool m_show_rvd;
        bool m_show_points;
        vector<vec2> m_points;
        int m_num_points;
        int m_lloyd_maxit;
        int m_newton_maxit;
        Mesh m_domain;
        Mesh m_rvd_mesh;
        CentroidalVoronoiTesselation * m_cvt;
        RestrictedVoronoiDiagram_var m_rvd;
        Delaunay_var m_delaunay;
};

int main(int argc, char** argv) {
    CVTApplication app(argc, argv, "");
    app.start();
    return 0;
}
