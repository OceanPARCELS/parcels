---
html_theme.sidebar_secondary.remove: true
---

# 🎓 Parcels v4 migration guide

This migration guide gives some tips if you want to migrate your Parcels v3 code to Parcels v4. The biggest changes are in the [Kernel API](#kernels) and the way that [FieldSets are created](#fieldset). The other changes are mostly small and should be easy to fix.

<style>
.migration-chat {
    --migration-change-bg: light-dark(#eaf3ff, #1e3a5f);
    --migration-change-border: light-dark(#9fc3f6, #4a7ba7);
    --migration-how-bg: light-dark(#edfced, #1f3a1f);
    --migration-how-border: light-dark(#9fd9a6, #4a7d4a);
    --migration-header-change-bg: light-dark(#dcecff, #2d4a7a);
    --migration-header-change-border: light-dark(#7fa9eb, #5a8bc9);
    --migration-header-how-bg: light-dark(#dcf7dc, #2d5a2d);
    --migration-header-how-border: light-dark(#7fc48a, #5ab85a);
    --migration-divider: light-dark(#74777c, #8b8e93);
    --migration-radius: 10px;
    --migration-width: 85%;
}

.migration-chat .migration-bubble {
    border-radius: var(--migration-radius);
    max-width: var(--migration-width);
}

.migration-chat .migration-change {
    background-color: var(--migration-change-bg);
    border: 1px solid var(--migration-change-border);
    padding: 0.7em 0.9em;
    margin: 0.6em 0 0.4em 0;
}

.migration-chat .migration-how {
    background-color: var(--migration-how-bg);
    border: 1px solid var(--migration-how-border);
    padding: 0.7em 0.9em;
    margin: 0.2em 0 0.6em auto;
    text-align: right;
}

.migration-chat .migration-header {
    font-weight: 700;
    padding: 0.55em 0.9em;
}

.migration-chat .migration-header-change {
    background-color: var(--migration-header-change-bg);
    border: 1px solid var(--migration-header-change-border);
    margin: 0.6em 0 0.4em 0;
}

.migration-chat .migration-header-how {
    background-color: var(--migration-header-how-bg);
    border: 1px solid var(--migration-header-how-border);
    margin: 0.2em 0 0.8em auto;
    text-align: right;
}

.migration-chat .migration-divider {
    border: none;
    border-top: 2px solid var(--migration-divider);
    margin: 0.8em 0 1em 0;
}

@media (max-width: 768px) {
    .migration-chat {
        --migration-width: 100%;
    }
}
</style>

<div class="migration-chat">

<div class="migration-bubble migration-header migration-header-change">
Description of change
</div>
<div class="migration-bubble migration-header migration-header-how">
How to migrate
</div>

</div>

## Kernels

<div class="migration-chat">

<div class="migration-bubble migration-change">
The Kernel API has had some major changes (see also below)
</div>
<div class="migration-bubble migration-how">
Add the <code>@parcels.validate_kernel</code> decorator above your Kernel function definition to check for errors and warnings.
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
The Kernel loop is 'vectorized': the input of a Kernel is a collection of particles
</div>
<div class="migration-bubble migration-how">
Replace <code>if</code>-statements with <code>numpy.where</code> statements or <a href="examples/tutorial_Argofloats.html">boolean indexing</a>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
Functions should work on a collection of particles instead of a single particle
</div>
<div class="migration-bubble migration-how">
Use <code>numpy</code> functions instead of <code>math</code> functions
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
Functions shouldn't be converted to <code>Kernel</code> objects in a <code>pset.execute()</code> call
</div>
<div class="migration-bubble migration-how">
Simply pass the function(s) as a list to <code>pset.execute()</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
The <code>time</code> argument in the Kernel signature is removed
</div>
<div class="migration-bubble migration-how">
Use <code>particle.t</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>particle.lon</code>, <code>particle.lat</code> and <code>particle.depth</code> are renamed
</div>
<div class="migration-bubble migration-how">
Use <code>particles.x</code>, <code>particles.y</code> and <code>particles.z</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>particle_dlon</code> <code>particle_dlat</code> and <code>particle_ddepth</code> are renamed
</div>
<div class="migration-bubble migration-how">
Use <code>particles.dx</code>, <code>particles.dy</code> and <code>particles.dz</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
The <code>particle</code> argument in the Kernel signature is renamed to <code>particles</code>
</div>
<div class="migration-bubble migration-how">
Change the argument name in your Kernel signature to <code>particles</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>particle.delete()</code> is no longer valid
</div>
<div class="migration-bubble migration-how">
Use <code>StatusCode.Delete</code> in combination with boolean indexing - as detailed <a href="examples/tutorial_statuscodes.html">here</a>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
Kernels are not concatenated under the hood, so can't access each others variables or states
</div>
<div class="migration-bubble migration-how">
Use fieldset.context or particle data to share information between kernels
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
The <code>InteractionKernel</code> class is removed as normal Kernels now have access to <em>all</em> particles
</div>
<div class="migration-bubble migration-how">
Particle-particle interaction can be <a href="examples/tutorial_interaction.html">performed within normal Kernels</a>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
Automatic conversion from depth to sigma grids under the hood is removed
</div>
<div class="migration-bubble migration-how">
Explicitly use <code>convert_z_to_sigma_croco</code> in sampling kernels (such as the <code>AdvectionRK4_3D_CROCO</code> or <code>SampleOMegaCroco</code> kernels) when working with <a href="examples/tutorial_croco_3D">CROCO data</a>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
The default advection scheme is changed from RK4 to RK2 as it is <a href="examples/tutorial_dt_integrators.html">faster while the accuracy is comparable for most applications</a>
</div>
<div class="migration-bubble migration-how">
Use <code>pset.execute(parcels.AdvectionRK2)</code> for advection
</div>
<hr class="migration-divider" />

</div>

## FieldSet

<div class="migration-chat">

<div class="migration-bubble migration-change">
<code>FieldSet.from_&lt;modelname&gt;()</code> is removed
</div>
<div class="migration-bubble migration-how">
Convert the model data first to <a href="https://sgrid.github.io/sgrid/">SGrid</a> (for example with a <code>parcels.covert_&lt;MODEL&gt;_to_sgrid()</code> function) and then use <code>FieldSet.from_sgrid_conventions()</code>, as also described in the <a href="getting_started/tutorial_quickstart.html#input-flow-fields-fieldset">Quickstart tutorial</a>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>FieldSet.interp_method</code> doesn't accept a string (e.g. <code>"linear"</code> or <code>"nearest"</code>)
</div>
<div class="migration-bubble migration-how">
Use an Interpolation function such as <code>parcels.interpolators.Linear</code> or <code>parcels.interpolators.Nearest</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>FieldSet.add_constant</code> is removed to reflect that this value no longer has to be constant
</div>
<div class="migration-bubble migration-how">
Use <code>FieldSet.add_context</code> to add a context variable to the FieldSet
</div>
<hr class="migration-divider" />

</div>

## Particle

<div class="migration-chat">

<div class="migration-bubble migration-change">
<code>Particle.add_variable()</code> is removed
</div>
<div class="migration-bubble migration-how">
Use <code>Particle.add_variables()</code>, which also takes a list of <code>Variables</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
Parameters <code>indices</code>, <code>timestamps</code>, and <code>time_periodic</code> removed from <code>FieldSet</code> constructors.
</div>
<div class="migration-bubble migration-how">
Simply manipulate the datasets with Xarray to the same effect before passing them to Parcels.
</div>
<hr class="migration-divider" />

</div>

## ParticleSet

<div class="migration-chat">

<div class="migration-bubble migration-change">
<code>pset.execute()</code> does not require Kernel objects
</div>
<div class="migration-bubble migration-how">
Simply pass the function(s) as a list to <code>pset.execute()</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>repeatdt</code> is removed
</div>
<div class="migration-bubble migration-how">
See the <a href="examples/tutorial_delaystart.html#release-particles-repeatedly">Delayed starts tutorial</a> for how to implement repeated releases of particles
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>lonlatdepth_dtype</code> is removed
</div>
<div class="migration-bubble migration-how">
Set the dtype of your particle variables directly
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
ParticleSet.execute() expects <code>datatime</code>, <code>numpy.datetime64</code> or <code>numpy.timedelta.64</code> for <code>runtime</code> and <code>endtime</code>
</div>
<div class="migration-bubble migration-how">
Update <code>runtime</code> and <code>endtime</code> to use <code>numpy.datetime64</code> or <code>numpy.timedelta.64</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>ParticleSet.from_field()</code>, <code>ParticleSet.from_line()</code>, <code>ParticleSet.from_list()</code> are removed
</div>
<div class="migration-bubble migration-how">
Use the <a href="examples/tutorial_delaystart.html"><code>ParticleSet</code> constructor</a> directly
</div>
<hr class="migration-divider" />

</div>

## ParticleFile

<div class="migration-chat">

<div class="migration-bubble migration-change">
ParticleFiles output is in <a href="getting_started/tutorial_output.html">parquet format</a>
</div>
<div class="migration-bubble migration-how">
Read the output with <code>polars.read_parquet()</code> or (to automatically handle cftime) <code>parcels.read_particlefile()</code>. For compatility with old postprocessing codes, convert the new Parquet output to v3-style zarr output using <code>parcels.particlefile_to_v3_zarr()</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>ParticleFile</code> is not a method of the <code>ParticleSet</code> class
</div>
<div class="migration-bubble migration-how">
Use <code>ParticleFile(...)</code> instead of <code>pset.ParticleFile(...)</code>
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>ParticleFile</code> errors out if there's existing output
</div>
<div class="migration-bubble migration-how">
Remove the existing output file or add <code>mode="w"</code> to overwrite
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
The output file does not have a <code>trajectory</code> dimension
</div>
<div class="migration-bubble migration-how">
Use <code>particle_id</code> instead of <code>trajectory</code> in your code
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>ParticleFile</code> does not have a <code>name</code> attribute
</div>
<div class="migration-bubble migration-how">
Use <code>ParticleFile.path</code>, which can be a string or a Path
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>ParticleFile</code> does not have a <code>chunks</code> argument
</div>
<div class="migration-bubble migration-how">
Remove the <code>chunks</code> argument from your code
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>ParticleFile</code> does not have a <code>to_write="once"</code> argument
</div>
<div class="migration-bubble migration-how">
Either write all timesteps of a variable, or none
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
Particles are not written when they are deleted
</div>
<div class="migration-bubble migration-how">
Call <code>ParticleFile.write()</code> <a href="examples/tutorial_write_in_kernel.html#writing-on-particle-deletion">inside a kernel to write out</a> deleted particles
</div>
<hr class="migration-divider" />

</div>

## Field

<div class="migration-chat">

<div class="migration-bubble migration-change">
Calling <code>Field.eval()</code> directly gives a warning if any values are out of bounds
</div>
<div class="migration-bubble migration-how">
Use <code>Field.eval()</code> as before, but check for warnings
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>Field.eval()</code> returns an array of floats (related to the vectorization)
</div>
<div class="migration-bubble migration-how">
Use <code>Field.eval()</code> as before, but expect an array of floats instead of a single float
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
The <code>NestedField</code> class is removed
</div>
<div class="migration-bubble migration-how">
See the <a href="examples/tutorial_nestedgrids.html">Nested Grids tutorial</a> for how to set up Nested Grids in v4
</div>
<hr class="migration-divider" />

<div class="migration-bubble migration-change">
<code>applyConversion</code> is removed
</div>
<div class="migration-bubble migration-how">
Interpolation on VectorFields automatically converts from m/s to degrees/s for spherical meshes. Other conversion of units should be handled in Interpolators or Kernels
</div>
<hr class="migration-divider" />

</div>

## GridSet

<div class="migration-chat">

<div class="migration-bubble migration-change">
<code>GridSet</code> is a list
</div>
<div class="migration-bubble migration-how">
Change <code>fieldset.gridset.grids[0]</code> to <code>fieldset.gridset[0]</code>
</div>
<hr class="migration-divider" />

</div>

## UnitConverters

<div class="migration-chat">

<div class="migration-bubble migration-change">
The <code>UnitConverter</code> class is removed
</div>
<div class="migration-bubble migration-how">
Remove any <code>UnitConverter</code> usage from your code and Interpolation functions should handle unit conversion internally, based on the value of <code>grid._mesh</code> (<code>"spherical"</code> or <code>"flat"</code>)
</div>
<hr class="migration-divider" />

</div>
