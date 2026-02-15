// Simple Vector Dot Product - TESTED VERSION
module vector_dot_product 
    import microgpt_pkg::*;
#(
    parameter int VEC_LEN = 16
) 
(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  fixed_t      vec_a [VEC_LEN-1:0],
    input  fixed_t      vec_b [VEC_LEN-1:0],
    output fixed_t      result,
    output logic        valid
);

    typedef enum logic [1:0] {
        IDLE,
        MULTIPLY,
        ACCUMULATE,
        DONE
    } state_t;
    
    state_t state;
    
    fixed_t products [VEC_LEN-1:0];
    fixed_t accumulator;
    int idx;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            accumulator <= '0;
            idx <= 0;
            valid <= 1'b0;
            for (int i = 0; i < VEC_LEN; i++) begin
                products[i] <= '0;
            end
        end else begin
            case (state)
                IDLE: begin
                    valid <= 1'b0;
                    if (start) begin
                        idx <= 0;
                        accumulator <= '0;
                        state <= MULTIPLY;
                    end
                end
                
                MULTIPLY: begin
                    // Compute all products
                    for (int i = 0; i < VEC_LEN; i++) begin
                        products[i] <= fixed_mul(vec_a[i], vec_b[i]);
                    end
                    state <= ACCUMULATE;
                end
                
                ACCUMULATE: begin
                    // Sum all products using blocking assignment
                    // The loop synthesizes to a combinational adder tree
                    fixed_t temp_sum;
                    temp_sum = '0;
                    for (int i = 0; i < VEC_LEN; i++) begin
                        temp_sum = fixed_add(temp_sum, products[i]);
                    end
                    accumulator <= temp_sum;
                    state <= DONE;
                end
                
                DONE: begin
                    valid <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
    assign result = accumulator;

endmodule : vector_dot_product