use nannou::prelude::*;
use nannou::{prelude::Update, window, App, Frame};
// use ode_solvers::dop_shared::IntegrationError;
use ode_solvers::dopri5::*;
// use ode_solvers::*;
type DestructuredState = DVector<f64>;
pub type State = DVector<f64>;
type Time = f64;

use nalgebra::{vector, DVector, Vector2};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;

use rayon;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    rope1: Rope,
    rope2: Rope,
    _window: window::Id,
}

fn model(app: &App) -> Model {
    let _window = app.new_window().view(view).build().unwrap();

    let my_rope = construct_rope(vector![0.0,0.0],200, 0.26, 307000.0, 25.0);
    let my_rope_2 = construct_rope(vector![0.0,0.0], 100, 0.26, 307000.0, 25.0);

    Model {
        rope1: my_rope,
        rope2: my_rope_2,
        _window,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    if app.mouse.buttons.left().is_down() {
        model.rope1.nodes[0].r =nannou_to_nalgebra(to_physics_coords(app.mouse.position(), app.window_rect()));
    }

    model.rope1.step();
    model.rope2.step();
}

fn view(app: &App, _model: &Model, frame: Frame) {
    // Prepare to draw.
    let draw = app.draw();

    // Clear the background to purple.
    draw.background().color(PLUM);

    let points = _model
        .rope1
        .nodes
        .iter()
        .map(|n| to_screen_coords(nalgebra_to_nannou(n.r), app.window_rect()))
        .clone();

    draw.path()
        .stroke()
        .weight(2.0)
        .points(points.clone())
        .color(STEELBLUE);

    for pnt in points.clone() {
        draw.ellipse().x_y(pnt.x, pnt.y).radius(1.0);
    }

    let points = _model
        .rope2
        .nodes
        .iter()
        .map(|n| to_screen_coords(nalgebra_to_nannou(n.r), app.window_rect()))
        .clone();

    draw.path()
        .stroke()
        .weight(2.0)
        .points(points.clone())
        .color(STEELBLUE);

    for pnt in points.clone() {
        draw.ellipse().x_y(pnt.x, pnt.y).radius(1.0);
    }

    draw.to_frame(app, &frame).unwrap();
}

fn construct_rope(origin: Vector2<f64>,num_segments: usize, mass: f64, spring_constant: f64, length: f64) -> Rope {
    /*creates a rope object with the given values. The rope is initially un-stretched and has one end at (0,0) and the other at (length,0) */

    // construct the rope object to be populated with nodes and relationships
    let mut rope = Rope {
        nodes: vec![],
        neighbors: HashMap::new(),
        // K: spring_constant,
        k: spring_constant*(num_segments as f64),
        // M: mass,
        m: mass / ((num_segments + 1) as f64),
        l: length,
    };

    // construct the list of nodes that will be in the rope
    {
        let mut nodes = vec![];

        for node_index in 0..num_segments + 1 {
            let new_node = Node {
                // id: node_index,
                r: origin+vector![
                    (rope.l) * (node_index as f64) / ((num_segments) as f64),
                    0.0
                    
                ],
                v: vector![0.0, 0.0],
            };
            nodes.push(new_node);
        }

        rope.nodes = nodes;
    }

    //construct the HashMap of neighbors for the rope
    {
        let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();

        for node_index in 0..num_segments + 1 {
            let mut ns_neighbor_indices = vec![];

            if node_index > 0 {
                ns_neighbor_indices.push(node_index - 1);
                if node_index < num_segments {
                    ns_neighbor_indices.push(node_index + 1);
                }
            }

            neighbors.insert(node_index, ns_neighbor_indices);
        }

        rope.neighbors = neighbors;
    }

    rope
}

#[derive(Copy, Clone, Debug)]
struct Node {
    // id: usize,
    r: Vector2<f64>,
    v: Vector2<f64>,
    
}

#[derive(Clone, Debug)]
struct Rope {
    l: f64,
    // M: f64,
    m: f64,
    // K: f64,
    k: f64,
    nodes: Vec<Node>,
    neighbors: HashMap<usize, Vec<usize>>,
}

impl Rope {
    fn step(&mut self) {
        let new_data = self.next_destructured_state();
        self.update_from_destructured_state(new_data);
    }
    fn segment_length(&self) -> f64 {
        self.l / ((self.nodes.len() - 1) as f64)
    }
    fn destructured_state(&self) -> DestructuredState {
        let mut current_state: DestructuredState = DVector::from_vec(vec![]);

        for node in &self.nodes {
            current_state.extend(node.r.iter().copied());
            current_state.extend(node.v.iter().copied());
        }

        current_state
    }
    fn update_from_destructured_state(&mut self, destructured_state: DestructuredState) {
        let num_nodes = self.nodes.len();
        let state = destructured_state;
        assert_eq!(
            num_nodes * 4,
            state.len(),
            "The given state is the wrong length for this rope"
        );

        for particle_index in 0..num_nodes {
            let state_index = 4 * particle_index;
            self.nodes[particle_index].r.x = state[state_index];
            self.nodes[particle_index].r.y = state[state_index + 1];
            self.nodes[particle_index].v.x = state[state_index + 2];
            self.nodes[particle_index].v.y = state[state_index + 3];
        }
    }
    fn next_destructured_state(&self) -> DestructuredState {
        let mut stepper = Dopri5::new(
            self,
            0.0,
            0.001,
            0.001,
            self.destructured_state(),
            1.0e-5,
            1.0e-5,
        );

        _ = stepper.integrate();

        (*stepper.y_out()).last().unwrap().clone()
    }
}

impl ode_solvers::System<State> for &Rope {
    // Equations of motion of the system
    fn system(&self, _t: Time, state: &State, delta_state: &mut State) {
        (1u64..((self.nodes.len()-1) as u64)).into_par_iter().for_each(|particle_index| {});
        
        for particle_index in 1..(self.nodes.len()-1) {
            let state_index = 4 * particle_index;

            let r = vector![state[state_index], state[state_index + 1]];

            let mut force = vector![0.0, 0.0];

            // calculate spring force from each neighbor
            for neighbor_index in &self.neighbors[&particle_index] {
                let neighbor_state_index = neighbor_index * 4;
                let other_r = vector![state[neighbor_state_index], state[neighbor_state_index + 1]];
                // let other_r = vector![0.0, 0.0];
                let relative_position = r - other_r;

                let sep = relative_position.magnitude();
                let direction = relative_position.normalize();

                // println!("{},{}", r.y, other_r.y);
                force += -self.k * (sep - self.segment_length()) * direction;
            }

            // other forces can be added here (gravity, air resistance, etc.)
            force += vector![0.0, -980.0];

            force -= 1.0*vector![state[state_index + 2],state[state_index + 3]];

            let acceleration = force/ self.m;

            // now we update the state
            // x velocity
            delta_state[state_index + 2] = acceleration.x;

            // y velocity
            delta_state[state_index + 3] = acceleration.y;

            // x position
            delta_state[state_index] = state[state_index + 2];

            // y position
            delta_state[state_index + 1] = state[state_index + 3];
        }
    }
}

#[allow(dead_code)]
pub fn to_screen_coords(pos: Vec2, window: Rect) -> Vec2 {
    let window_height = 100.0 * window.h() / window.w();
    let x = map_range(
        pos.x,
        -100.0 / 2.0,
        100.0 / 2.0,
        window.left(),
        window.right(),
    );
    let y = map_range(
        pos.y,
        -window_height / 2.0,
        window_height / 2.0,
        window.bottom(),
        window.top(),
    );
    pt2(x, y)
}

#[allow(dead_code)]
pub fn to_physics_coords(pos: Vec2, window: Rect) -> Vec2 {
    let window_height = 100.0 * window.h() / window.w();
    let x = map_range(
        pos.x,
        window.left(),
        window.right(),
        -100.0 / 2.0,
        100.0 / 2.0,
    );
    let y = map_range(
        pos.y,
        window.bottom(),
        window.top(),
        -window_height / 2.0,
        window_height / 2.0,
    );
    pt2(x, y)
}

fn nannou_to_nalgebra(vec: Vec2) -> nalgebra::Vector2<f64> {
    nalgebra::Vector2::new(vec.x as f64, vec.y as f64)
}

fn nalgebra_to_nannou(vec: nalgebra::Vector2<f64>) -> Vec2 {
    pt2(vec.x as f32, vec.y as f32)
}
